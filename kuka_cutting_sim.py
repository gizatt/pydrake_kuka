# -*- coding: utf8 -*-

import argparse
import random
import time
import os

import matplotlib.pyplot as plt
import numpy as np

import pydrake
from pydrake.all import (
    CompliantMaterial,
    DiagramBuilder,
    RigidBodyFrame,
    RigidBodyPlant,
    RigidBodyTree,
    RungeKutta2Integrator,
    RungeKutta3Integrator,
    Shape,
    SignalLogger,
    Simulator,
)

from underactuated.meshcat_rigid_body_visualizer import (
    MeshcatRigidBodyVisualizer)

import kuka_controllers
import kuka_ik
import kuka_utils
import cutting_utils

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", "--duration",
                        type=float,
                        help="Duration to run sim.",
                        default=1000.0)
    parser.add_argument("--test",
                        action="store_true",
                        help="Help out CI by launching a meshcat server for "
                             "the duration of the test.")
    parser.add_argument("-N", "--n_objects",
                        type=int, default=2,
                        help="# of objects to spawn")
    parser.add_argument("--seed",
                        type=float, default=time.time(),
                        help="RNG seed")
    parser.add_argument("--animate_forever",
                        action="store_true",
                        help="Animates the completed sim in meshcat repeatedly.")

    args = parser.parse_args()
    int_seed = int(args.seed*1000. % 2**32)
    random.seed(int_seed)
    np.random.seed(int_seed)

    meshcat_server_p = None
    if args.test:
        print "Spawning"
        import subprocess
        meshcat_server_p = subprocess.Popen(["meshcat-server"])
    else:
        print "Warning: if you have not yet run meshcat-server in another " \
              "terminal, this will hang."

    # Construct the initial robot and its environment
    world_builder = kuka_utils.ExperimentWorldBuilder()

    rbt, rbt_just_kuka, q0 = world_builder.setup_initial_world(
        n_objects = args.n_objects)
    x = np.zeros(rbt.get_num_positions() + rbt.get_num_velocities())
    x[0:q0.shape[0]] = q0
    t = 0
    while 1:
        mrbv = MeshcatRigidBodyVisualizer(rbt, draw_timestep=0.01)
        # (wait while the visualizer warms up and loads in the models)
        mrbv.draw(x)

        # Make our RBT into a plant for simulation
        rbplant = RigidBodyPlant(rbt)
        allmaterials = CompliantMaterial()
        allmaterials.set_youngs_modulus(1E8)  # default 1E9
        allmaterials.set_dissipation(0.8)     # default 0.32
        allmaterials.set_friction(0.9)        # default 0.9.
        rbplant.set_default_compliant_material(allmaterials)

        # Build up our simulation by spawning controllers and loggers
        # and connecting them to our plant.
        builder = DiagramBuilder()
        # The diagram takes ownership of all systems
        # placed into it.
        rbplant_sys = builder.AddSystem(rbplant)


        # Spawn the controller that drives the Kuka to its
        # desired posture.
        kuka_controller = builder.AddSystem(
            kuka_controllers.KukaController(rbt, rbplant_sys))
        builder.Connect(rbplant_sys.state_output_port(),
                        kuka_controller.robot_state_input_port)
        builder.Connect(kuka_controller.get_output_port(0),
                        rbplant_sys.get_input_port(0))

        # Same for the hand
        hand_controller = builder.AddSystem(
            kuka_controllers.HandController(rbt, rbplant_sys))
        builder.Connect(rbplant_sys.state_output_port(),
                        hand_controller.robot_state_input_port)
        builder.Connect(hand_controller.get_output_port(0),
                        rbplant_sys.get_input_port(1))

        # And the guillotine
        knife_controller = builder.AddSystem(
            kuka_controllers.GuillotineController(rbt, rbplant_sys))
        builder.Connect(rbplant_sys.state_output_port(),
                        knife_controller.robot_state_input_port)
        builder.Connect(knife_controller.get_output_port(0),
                        rbplant_sys.get_input_port(2))

        # Create a high-level state machine to guide the robot
        # motion...
        manip_state_machine = builder.AddSystem(
            kuka_controllers.ManipStateMachine(
                rbt, rbt_just_kuka, x[0:7],
                world_builder=world_builder,
                hand_controller=hand_controller,
                kuka_controller=kuka_controller,
                mrbv = mrbv))
        builder.Connect(rbplant_sys.state_output_port(),
                        manip_state_machine.robot_state_input_port)
        builder.Connect(manip_state_machine.hand_setpoint_output_port,
                        hand_controller.setpoint_input_port)
        builder.Connect(manip_state_machine.kuka_setpoint_output_port,
                        kuka_controller.setpoint_input_port)

        cutting_guard = builder.AddSystem(
            cutting_utils.CuttingGuard(
                name="blade cut guard",
                rbt=rbt, rbp=rbplant,
                cutting_body_index=world_builder.guillotine_blade_index,
                cut_direction=[0., 0., -1.],
                cut_normal=[1., 0., 0.],
                min_cut_force=10.,
                cuttable_body_indices=world_builder.manipuland_body_indices,
                timestep=0.001,
                last_cut_time = t))
        builder.Connect(rbplant_sys.state_output_port(),
                        cutting_guard.state_input_port)
        builder.Connect(rbplant_sys.contact_results_output_port(),
                        cutting_guard.contact_results_input_port)
        
        # Hook up loggers for the robot state, the robot setpoints,
        # and the torque inputs.
        def log_output(output_port, rate):
            logger = builder.AddSystem(SignalLogger(output_port.size()))
            logger._DeclarePeriodicPublish(1. / rate, 0.0)
            builder.Connect(output_port, logger.get_input_port(0))
            return logger
        state_log = log_output(rbplant_sys.get_output_port(0), 60.)
        setpoint_log = log_output(
            manip_state_machine.kuka_setpoint_output_port, 60.)
        kuka_control_log = log_output(
            kuka_controller.get_output_port(0), 60.)

        # Hook up the visualizer we created earlier.
        visualizer = builder.AddSystem(mrbv)
        builder.Connect(rbplant_sys.state_output_port(),
                        visualizer.get_input_port(0))

        # Done! Compile it all together and visualize it.
        diagram = builder.Build()

        # Create a simulator for it.
        simulator = Simulator(diagram)
        simulator.Initialize()
        simulator.set_target_realtime_rate(1.0)
        # Simulator time steps will be very small, so don't
        # force the rest of the system to update every single time.
        simulator.set_publish_every_time_step(False)


        # From iiwa_wsg_simulation.cc:
        # When using the default RK3 integrator, the simulation stops
        # advancing once the gripper grasps the box.  Grasping makes the
        # problem computationally stiff, which brings the default RK3
        # integrator to its knees.
        timestep = 0.00005
        integrator = RungeKutta2Integrator(diagram, timestep,
                                           simulator.get_mutable_context())
        simulator.reset_integrator(integrator)

        # The simulator simulates forward from a given Context,
        # so we adjust the simulator's initial Context to set up
        # the initial state.
        state = simulator.get_mutable_context().\
            get_mutable_continuous_state_vector()
        initial_state = np.zeros(x.shape)
        initial_state[0:x.shape[0]] = x.copy()
        state.SetFromVector(initial_state)
        simulator.get_mutable_context().set_time(t)

        # This kicks off simulation. Most of the run time will be spent
        # in this call.
        try:
            simulator.StepTo(args.duration)
        except cutting_utils.CutException as e:
            t = simulator.get_mutable_context().get_time()
            print "Handling cut event at time %f" % t
            x = simulator.get_mutable_context().\
                get_mutable_continuous_state_vector().CopyToVector()[0:x.shape[0]]
            rbt, x = world_builder.do_cut(
                rbt, x, cut_body_index=e.cut_body_index, 
                cut_pt=e.cut_pt, cut_normal=e.cut_normal)
            continue
        except StopIteration:
            print "Terminated early"
        break

    print("Final state: ", state.CopyToVector())
    end_time = simulator.get_mutable_context().get_time()

    if args.animate_forever:
        try:
            while(1):
                mrbv.animate(state_log, time_scaling=0.1)
        except Exception as e:
            print "Fail ", e

    if meshcat_server_p is not None:
        meshcat_server_p.kill()
        meshcat_server_p.wait()
