# -*- coding: utf8 -*-

import argparse
import random
import time
import os

import matplotlib.pyplot as plt
import numpy as np

import pydrake
from pydrake.all import (
    DiagramBuilder,
    RgbdCamera,
    RigidBodyFrame,
    RigidBodyPlant,
    RigidBodyTree,
    RungeKutta2Integrator,
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

    # Construct the robot and its environment
    rbt = RigidBodyTree()
    world_builder = kuka_utils.ExperimentWorldBuilder()
    world_builder.setup_kuka(rbt)
    z_table = world_builder.table_top_z_in_world
    rbt_just_kuka = rbt.Clone()
    for k in range(args.n_objects):
        world_builder.add_cut_cylinder_to_tabletop(rbt, "cyl_%d" % k)
    rbt.compile()
    rbt_just_kuka.compile()

    # Figure out initial pose for the arm
    ee_body=rbt_just_kuka.FindBody("right_finger").get_body_index()
    ee_point=np.array([0.0, 0.03, 0.0])
    end_effector_desired = np.array([0.5, 0.0, z_table+0.5, -np.pi/2., 0., 0.])
    q0_kuka_seed = rbt_just_kuka.getZeroConfiguration()
    # "Center low" from IIWA stored_poses.json from Spartan
    # + closed hand
    q0_kuka_seed[0:7] = np.array([-0.18, -1., 0.12, -1.89, 0.1, 1.3, 0.38])
    q0_kuka, info = kuka_ik.plan_ee_configuration(
        rbt_just_kuka, q0_kuka_seed, q0_kuka_seed, end_effector_desired, ee_body,
        ee_point, allow_collision=True, euler_limits=0.01)
    if info != 1:
        print "Info %d on IK for initial posture." % info
    q0 = rbt.getZeroConfiguration()
    q0[0:9] = q0_kuka
    q0 = world_builder.project_rbt_to_nearest_feasible_on_table(
        rbt, q0)

    mrbv = MeshcatRigidBodyVisualizer(rbt, draw_timestep=0.01)
    # (wait while the visualizer warms up and loads in the models)
    time.sleep(0.25)

    # Make our RBT into a plant for simulation
    rbplant = RigidBodyPlant(rbt)

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

    # Create a high-level state machine to guide the robot
    # motion...
    manip_state_machine = builder.AddSystem(
        kuka_controllers.ManipStateMachine(
            rbt, rbt_just_kuka, q0[0:7],
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
            rbt=rbt, rbp=rbplant,
            cutting_body_index=rbt.FindBody("right_finger").get_body_index(),
            cut_direction=[0., 1., 0.],
            min_cut_force=1.,
            cuttable_body_indices=world_builder.manipuland_body_indices,
            timestep=0.01))
    builder.Connect(rbplant_sys.state_output_port(),
                    cutting_guard.state_input_port)
    builder.Connect(rbplant_sys.contact_results_output_port(),
                    cutting_guard.contact_results_input_port)
    
    # Hook up the visualizer we created earlier.
    visualizer = builder.AddSystem(mrbv)
    builder.Connect(rbplant_sys.state_output_port(),
                    visualizer.get_input_port(0))

    # Done! Compile it all together and visualize it.
    diagram = builder.Build()
    kuka_utils.render_system_with_graphviz(diagram, "view.gv")

    # Create a simulator for it.
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    # Simulator time steps will be very small, so don't
    # force the rest of the system to update every single time.
    simulator.set_publish_every_time_step(False)

    # The simulator simulates forward from a given Context,
    # so we adjust the simulator's initial Context to set up
    # the initial state.
    state = simulator.get_mutable_context().\
        get_mutable_continuous_state_vector()
    initial_state = np.zeros(state.size())
    initial_state[0:q0.shape[0]] = q0
    state.SetFromVector(initial_state)

    # From iiwa_wsg_simulation.cc:
    # When using the default RK3 integrator, the simulation stops
    # advancing once the gripper grasps the box.  Grasping makes the
    # problem computationally stiff, which brings the default RK3
    # integrator to its knees.
    timestep = 0.0001
    simulator.reset_integrator(
        RungeKutta2Integrator(diagram, timestep,
                              simulator.get_mutable_context()))

    # This kicks off simulation. Most of the run time will be spent
    # in this call.
    try:
        simulator.StepTo(args.duration)
    except StopIteration:
        print "Terminated early"
    print("Final state: ", state.CopyToVector())
    end_time = simulator.get_mutable_context().get_time()

    if meshcat_server_p is not None:
        meshcat_server_p.kill()
        meshcat_server_p.wait()
