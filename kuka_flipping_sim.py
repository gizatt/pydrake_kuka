# -*- coding: utf8 -*-

import argparse
from copy import deepcopy
import random
import time
import os

import matplotlib.pyplot as plt
import numpy as np

import pydrake
from pydrake.all import (
    CompliantMaterial,
    DiagramBuilder,
    PiecewisePolynomial,
    RigidBodyFrame,
    RigidBodyPlant,
    RigidBodyTree,
    RungeKutta2Integrator,
    RungeKutta3Integrator,
    Shape,
    SignalLogger,
    Simulator,
)

from underactuated import MeshcatRigidBodyVisualizer
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
                        default=3.0)
    parser.add_argument("--test",
                        action="store_true",
                        help="Help out CI by launching a meshcat server for "
                             "the duration of the test.")
    parser.add_argument("--seed",
                        type=float, default=0,
                        help="RNG seed")
    parser.add_argument("--animate_forever",
                        action="store_true",
                        help="Animates the completed sim in meshcat on loop.")

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

    pose_base = np.array([0.65, 0., 0.75, np.pi/2., -np.pi/2., 0.])

    for k in range(5):
        # Construct the initial robot and its environment
        world_builder = kuka_utils.ExperimentWorldBuilder(with_knife=False)

        # These are the initial state of the sim that is relevant to the task:
        # Functionally, I'm thinking of these as "start at a nearly-fixed
        # position relative to the object" with varying object size and some
        # noise on its pose estimate (which comes in as object pose variation)
        pose_perturb = pose_base.copy()
        pose_perturb[0:2] += (np.random.random(2)-0.5)*0.1
        pose_perturb[3:6] += (np.random.random(3))*np.pi/4.
        height = 0.1 + np.random.random() * 0.05
        radius = 0.02 + np.random.random() * 0.01
        rbt, rbt_just_kuka, q0 = world_builder.setup_initial_world(
            n_objects=1, cylinder_poses=[pose_perturb],
            cylinder_cut_dirs=[[[1., 0., 0.]]],
            cylinder_cut_points=[[[0., 0., 0.]]],
            cylinder_radiuses=[radius],
            cylinder_heights=[height])
        rbt.compile()
        x = np.zeros(rbt.get_num_positions() + rbt.get_num_velocities())
        x[0:q0.shape[0]] = q0
        t = 0
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
            kuka_controllers.InstantaneousKukaController(rbt, rbplant_sys))
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

        flipper_params = kuka_controllers.DynamicFlipPrimitiveParameters(rbt)
        # Default / semi-reasonable values for the policy:
        # See DynamicFlipPrimitiveParameters in kuka_controllers.py
        # for an explanation of these values
        flipper_params.table_touch_z_threshold = 0.8
        flipper_params.reach_object_time = 0.3
        flipper_params.end_flip_time = 0.5
        flipper_params.fingertip_bury_amount = 0.01
        flipper_params.object_overshoot_distance = 0.05
        flipper_params.followthrough_distance = 0.3
        flipper_params.followthrough_height = 0.3

        # Create a high-level state machine to guide the robot
        # motion...
        task_planner = builder.AddSystem(
            kuka_controllers.TaskPlannerOnlyFlipping(
                rbt, q0, world_builder,
                object_id=rbt.get_num_bodies()-1,
                params=flipper_params))
        builder.Connect(rbplant_sys.state_output_port(),
                        task_planner.robot_state_input_port)
        builder.Connect(task_planner.hand_setpoint_output_port,
                        hand_controller.setpoint_input_port)
        builder.Connect(task_planner.kuka_setpoint_output_port,
                        kuka_controller.setpoint_input_port)

        # Create a guard to end the sim when everything is
        # flipped.
        done_guard = builder.AddSystem(
            kuka_utils.AllFlippedGuard(rbt, world_builder))
        builder.Connect(rbplant_sys.state_output_port(),
                        done_guard.robot_state_input_port)

        # Hook up loggers for the robot state, the robot setpoints,
        # and the torque inputs.
        def log_output(output_port, rate):
            logger = builder.AddSystem(SignalLogger(output_port.size()))
            logger._DeclarePeriodicPublish(1. / rate, 0.0)
            builder.Connect(output_port, logger.get_input_port(0))
            return logger
        state_log = log_output(rbplant_sys.get_output_port(0), 60.)
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
        timestep = 0.0005
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

        success = False
        try:
            simulator.StepTo(args.duration)
        except kuka_utils.DoneException:
            print "Terminated early and succeeded"
            success = True
        except StopIteration:
            print "Terminated early but did not succeed"
        except RuntimeError as e:
            print "Runtime Error: ", e
            print "Probably NAN in simulation. Terminating early."

        print("Final state: ", state.CopyToVector())
        print("SUCCESS: ", success)
        end_time = simulator.get_mutable_context().get_time()

        if args.animate_forever:
            try:
                while (1):
                    mrbv.animate(state_log)
            except Exception as e:
                print "Exception during visualization: ", e

    if meshcat_server_p is not None:
        meshcat_server_p.kill()
        meshcat_server_p.wait()
