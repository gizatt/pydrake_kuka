# -*- coding: utf8 -*-
import math
import numpy as np
import random
import time

import pydrake
from pydrake.all import (
    BasicVector,
    LeafSystem,
    PiecewisePolynomial,
    PortDataType,
)

import kuka_ik
import kuka_utils

from underactuated import MeshcatRigidBodyVisualizer


class KukaController(LeafSystem):
    def __init__(self, rbt, plant,
                 control_period=0.005,
                 print_period=0.5):
        LeafSystem.__init__(self)
        self.set_name("Kuka Controller")

        self.controlled_joint_names = [
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7"
        ]

        self.controlled_inds, _ = kuka_utils.extract_position_indices(
            rbt, self.controlled_joint_names)
        # Extract the full-rank bit of B, and verify that it's full rank
        self.nq_reduced = len(self.controlled_inds)
        self.B = np.empty((self.nq_reduced, self.nq_reduced))
        for k in range(self.nq_reduced):
            for l in range(self.nq_reduced):
                self.B[k, l] = rbt.B[self.controlled_inds[k],
                                     self.controlled_inds[l]]
        if np.linalg.matrix_rank(self.B) < self.nq_reduced:
            print "The joint set specified is underactuated."
            sys.exit(-1)
        self.B_inv = np.linalg.inv(self.B)
        # Copy lots of stuff
        self.rbt = rbt
        self.nq = rbt.get_num_positions()
        self.plant = plant
        self.nu = plant.get_input_port(0).size()
        self.print_period = print_period
        self.last_print_time = -print_period
        self.shut_up = False

        self.robot_state_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   rbt.get_num_positions() +
                                   rbt.get_num_velocities())

        self.setpoint_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   rbt.get_num_positions() +
                                   rbt.get_num_velocities())

        self._DeclareDiscreteState(self.nu)
        self._DeclarePeriodicDiscreteUpdate(period_sec=control_period)
        self._DeclareVectorOutputPort(
            BasicVector(self.nu),
            self._DoCalcVectorOutput)

    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        # (This makes sure relevant event handlers get called.)
        LeafSystem._DoCalcDiscreteVariableUpdates(
            self, context, events, discrete_state)

        new_control_input = discrete_state. \
            get_mutable_vector().get_mutable_value()
        x = self.EvalVectorInput(
            context, self.robot_state_input_port.get_index()).get_value()
        x_des = self.EvalVectorInput(
            context, self.setpoint_input_port.get_index()).get_value()
        q = x[:self.nq]
        v = x[self.nq:]
        q_des = x_des[:self.nq]
        v_des = x_des[self.nq:]

        qerr = (q_des[self.controlled_inds] - q[self.controlled_inds])
        verr = (v_des[self.controlled_inds] - v[self.controlled_inds])

        kinsol = self.rbt.doKinematics(q, v)
        # Get the full LHS of the manipulator equations
        # given the current config and desired accelerations
        vd_des = np.zeros(self.rbt.get_num_positions())
        vd_des[self.controlled_inds] = 1000.*qerr + 100*verr
        lhs = self.rbt.inverseDynamics(kinsol, external_wrenches={}, vd=vd_des)
        new_u = self.B_inv.dot(lhs[self.controlled_inds])
        new_control_input[:] = new_u

    def _DoCalcVectorOutput(self, context, y_data):
        if (self.print_period and
                context.get_time() - self.last_print_time
                >= self.print_period):
            print "t: ", context.get_time()
            self.last_print_time = context.get_time()
        control_output = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        y[:] = control_output[:]


class HandController(LeafSystem):
    def __init__(self, rbt, plant,
                 control_period=0.001):
        LeafSystem.__init__(self)
        self.set_name("Hand Controller")

        self.controlled_joint_names = [
            "left_finger_sliding_joint",
            "right_finger_sliding_joint"
        ]

        self.max_force = 100.  # gripper max closing / opening force

        self.controlled_inds, _ = kuka_utils.extract_position_indices(
            rbt, self.controlled_joint_names)

        self.nu = plant.get_input_port(1).size()
        self.nq = rbt.get_num_positions()

        self.robot_state_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   rbt.get_num_positions() +
                                   rbt.get_num_velocities())

        self.setpoint_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   1)

        self._DeclareDiscreteState(self.nu)
        self._DeclarePeriodicDiscreteUpdate(period_sec=control_period)
        self._DeclareVectorOutputPort(
            BasicVector(self.nu),
            self._DoCalcVectorOutput)

    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        # (This makes sure relevant event handlers get called.)
        LeafSystem._DoCalcDiscreteVariableUpdates(
            self, context, events, discrete_state)

        new_control_input = discrete_state. \
            get_mutable_vector().get_mutable_value()
        x = self.EvalVectorInput(
            context, self.robot_state_input_port.get_index()).get_value()

        gripper_width_des = self.EvalVectorInput(
            context, self.setpoint_input_port.get_index()).get_value()

        q_full = x[:self.nq]
        v_full = x[self.nq:]

        q = q_full[self.controlled_inds]
        q_des = np.array([-gripper_width_des[0], gripper_width_des[0]])
        v = v_full[self.controlled_inds]
        v_des = np.zeros(2)

        qerr = q_des - q
        verr = v_des - v

        Kp = 1000.
        Kv = 100.
        new_control_input[:] = np.clip(
            Kp * qerr + Kv * verr, -self.max_force, self.max_force)

    def _DoCalcVectorOutput(self, context, y_data):
        control_output = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        y[:] = control_output[:]


class ManipStateMachine(LeafSystem):
    ''' Encodes the high-level logic
        for the manipulation system.

        It pulls the robotic to a nominal pose, and then
        (given as input a list of all of the cut cylinder
        poses) repeatedly tries to plan collision-free
        trajectories to a pose goal relative to a flippable
        edge.

        States:
           - Returning to nominal
           - Executing flip

        A flip plan is an open-loop joint trajectory that
        get the end effector to the object (stage 1), moves
        inwards towards the object until contact is made
        (stage 2), and then does an open-loop flip maneuver
        (stage 3).

        TODO in future: better control with contact:
        A "flip" plan comes with a couple of parts:
            - A joint trajectory to get the end effect
             close to the target point on the surface
            - The actual target point on the surface
              of both the gripper and the object, which
              we want to bring into contact
            - A pose trajectory of the object we want
              to achieve, along with a trajectory of
              the gripper

    '''

    STATE_STARTUP = -1
    STATE_MOVING_TO_NOMINAL = 0
    STATE_ATTEMPTING_FLIP = 1
    STATE_STUCK = 2

    def __init__(self, rbt_full, rbt_kuka, q_nom,
                 world_builder, hand_controller, kuka_controller,
                 mrbv):
        LeafSystem.__init__(self)
        self.set_name("Food Flipping State Machine")

        self.end_of_plan_time_margin = 0.1
        self.n_replan_attempts_nominal = 10
        self.n_replan_attempts_flip = 100

        self.q_nom = q_nom
        # TODO(gizatt) Move these into state vector?
        self.q_traj = None
        self.q_traj_d = None
        self.rbt_full = rbt_full
        self.rbt_kuka = rbt_kuka
        self.nq_full = rbt_full.get_num_positions()
        self.nq_kuka = rbt_kuka.get_num_positions()
        self.world_builder = world_builder
        self.hand_controller = hand_controller
        self.kuka_controller = kuka_controller
        self.mrbv = mrbv #MeshcatRigidBodyVisualizer(rbt_kuka, prefix="planviz")
    
        self.robot_state_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   rbt_full.get_num_positions() +
                                   rbt_full.get_num_velocities())

        self._DeclareDiscreteState(1)
        self._DeclarePeriodicDiscreteUpdate(period_sec=0.01)
        # TODO set default state somehow better. Requires new
        # bindings to override AllocateDiscreteState or something else.
        self.initialized = False

        # TODO The controller takes full state in, even though it only
        # controls the Kuka... that should be tidied up.
        self.kuka_setpoint_output_port = \
            self._DeclareVectorOutputPort(
                BasicVector(rbt_full.get_num_positions() +
                            rbt_full.get_num_velocities()),
                self._DoCalcKukaSetpointOutput)
        self.hand_setpoint_output_port = \
            self._DeclareVectorOutputPort(BasicVector(1),
                                          self._DoCalcHandSetpointOutput)

        self._DeclarePeriodicPublish(0.01, 0.0)

    def _PlanToNominal(self, q0, start_time):
        print "Searching for trajectory to get to nominal pose."
        success = False
        for k in range(self.n_replan_attempts_nominal):
            qtraj, info = kuka_ik.plan_trajectory_to_posture(
                self.rbt_full, q0, self.q_nom, 10, 0.5, start_time)
            if info not in [1, 3, 100]:
                print "Got error code, trying again."
                continue

            if kuka_utils.is_trajectory_collision_free(self.rbt_full, qtraj):
                success = True
                break
            else:
                print "Got a good error code, but trajectory was not " \
                      "collision-free. Trying again."
        if success:
            self.q_traj = qtraj
            self.q_traj_d = qtraj.derivative(1)
        return success

    def _PlanFlip(self, q0, start_time):
        # Pick an object
        print "Searching for trajectory to pick an object."
        success = False

        kinsol = self.rbt_full.doKinematics(q0)

        # Build set of flippable objects
        dirs = np.zeros((3, 4))
        dirs[:, 0] = [-1., 0., 0.]  # in direction of cut on cylinder
        dirs[:, 1] = [0., 1., 0.]  # in other non-axial direction of cylinder
        dirs[:, 2] = [0., 0., 1.]  # in other non-axial direction of cylinder
        dirs[:, 3] = [0., 0., 0.]  # origin
        # tuples of (reach pose, touch pose, flip pose)
        possible_flips = []
        for body_i in self.world_builder.manipuland_body_indices:
            # The -x dir in body frame is the cut dir, and it needs
            # to point up
            dirs_world = self.rbt_full.transformPoints(kinsol, dirs, body_i, 0)
            if dirs_world[2, 0] > dirs_world[2, 3] + 0.1:
                z = self.world_builder.table_top_z_in_world + 0.005
                dirs_world[2, 3] = z
                # touch moves along this direction in the world (xy)
                touch_dir = dirs_world[0:3, 3] - dirs_world[0:3, 1]
                touch_dir[2] = 0.
                if np.linalg.norm(touch_dir) == 0.:
                    touch_dir = dirs_world[0:3, 3] - dirs_world[0:3, 2]
                    touch_dir[2] = 0.
                touch_dir /= np.linalg.norm(touch_dir)

                # Try attacking in a couple of directions
                for t in np.linspace(-0.5, 0.5, 8):
                    rotation = np.array([[np.cos(t), -np.sin(t)],
                                         [np.sin(t), np.cos(t)]])
                    new_touch_dir = np.zeros(3)
                    new_touch_dir[0:2] = rotation.dot(touch_dir[0:2])
                    new_touch_dir[2] = touch_dir[2]
                    touch_yaw = math.atan2(new_touch_dir[1], new_touch_dir[0])
                    for m in [-1., 1.]:
                        for k in range(5):
                            rpy = [-np.pi/2., 0., m*touch_yaw]

                            touch_pose = np.hstack([dirs_world[:, 3], rpy])
                            reach_pose = np.hstack([dirs_world[:, 3] + new_touch_dir*m*0.1, rpy])
                            flip_pose = np.hstack([dirs_world[:, 3] - new_touch_dir*m*0.1, rpy])
                            flip_pose += [0., 0., 0.1, 0., 0., 0.]
                            possible_flips.append((reach_pose, touch_pose, flip_pose))

        if len(possible_flips) == 0:
            print "No possible flips! Done!"
            return False
        random.shuffle(possible_flips)

        collision_inds = []
        collision_inds += self.world_builder.manipuland_body_indices
        collision_inds.append(self.rbt_full.FindBody("right_finger").get_body_index())
        collision_inds.append(self.rbt_full.FindBody("left_finger").get_body_index())
        for k in range(self.rbt_full.get_num_bodies()):
            if self.rbt_full.get_body(k).get_name() == "link":
                collision_inds.append(k)
        ee_body=self.rbt_full.FindBody("right_finger").get_body_index()
        ee_point=[0.0, 0.03, 0.0]  # approximately fingertip
        for goals in possible_flips:
            q_reach, info = kuka_ik.plan_ee_configuration(
                self.rbt_full, q0, q0, ee_pose=goals[0], 
                ee_body=ee_body, ee_point=ee_point,
                allow_collision=False,
                active_bodies_idx=collision_inds)
            print "reach: ", info
            self.mrbv.draw(q_reach)
            time.sleep(0.)
            if info not in [1, 3, 100]:
                continue
            q_touch, info = kuka_ik.plan_ee_configuration(
                self.rbt_full, q_reach, q_reach, ee_pose=goals[1], 
                ee_body=ee_body, ee_point=ee_point,
                allow_collision=True,
                active_bodies_idx=collision_inds)
            print "touch: ", info
            self.mrbv.draw(q_touch)
            time.sleep(0.)
            if info not in [1, 3, 100]:
                continue
            q_flip, info = kuka_ik.plan_ee_configuration(
                self.rbt_full, q_touch, q_reach, ee_pose=goals[2], 
                ee_body=ee_body, ee_point=ee_point,
                allow_collision=False,
                active_bodies_idx=collision_inds)
            print "flip: ", info
            self.mrbv.draw(q_flip)
            time.sleep(0.)
            if info not in [1, 3, 100]:
                continue

            # If that checked out, make a real trajectory to hit all of those
            # points
            ts = np.array([0., 0.5, 0.75, 1.0])
            traj_seed = PiecewisePolynomial.Pchip(
                ts + start_time, np.vstack([q0, q_reach, q_touch, q_flip]).T, True)
            q_traj = traj_seed
            #needs more stable ik engine
            ##kuka_utils.visualize_plan_with_meshcat(self.rbt_kuka, self.mrbv, traj_seed)
            #q_traj, info, knots = kuka_ik.plan_ee_trajectory(
            #    self.rbt_full, q0, traj_seed,
            #    ee_times=ts[1:], ee_poses=goals,
            #    ee_body=ee_body, ee_point=ee_point,
            #    n_knots = 20,
            #    start_time = start_time,
            #    avoid_collision_tspans=[[ts[0], ts[1]]],
            #    active_bodies_idx=collision_inds)
            #for knot in knots:
            #    self.mrbv.draw(knot)
            #    time.sleep(0.25)
            # kuka_utils.visualize_plan_with_meshcat(self.rbt_kuka, self.mrbv, q_traj)

            if kuka_utils.is_trajectory_collision_free(self.rbt_full, q_traj):
                success = True
                break
            else:
                print "Got a good error code, but trajectory was not " \
                      "collision-free. Trying again."

        if success:
            self.q_traj = q_traj
            self.q_traj_d = q_traj.derivative(1)
        return success

    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        LeafSystem._DoCalcDiscreteVariableUpdates(
            self, context, events, discrete_state)

        state = discrete_state. \
            get_mutable_vector().get_mutable_value()
        if not self.initialized:
            state[0] = self.STATE_MOVING_TO_NOMINAL
            self.initialized = True

        t = context.get_time()
        x_robot_full = self.EvalVectorInput(
            context, self.robot_state_input_port.get_index()).get_value()
        q_robot_full = x_robot_full[0:self.nq_full]
        if self.q_traj:
            terminated = \
                t >= self.q_traj.end_time() + self.end_of_plan_time_margin
        else:
            terminated = True

        if state[0] == self.STATE_STARTUP or \
           (state[0] == self.STATE_ATTEMPTING_FLIP and terminated):
            if self._PlanToNominal(q_robot_full, t):
                state[0] = self.STATE_MOVING_TO_NOMINAL
                print "State machine moving to nominal"
            else:
                state[0] = self.STATE_STUCK
                self.q_traj = None
                self.q_traj_d = None
                print "State machine stuck"
        elif state[0] == self.STATE_MOVING_TO_NOMINAL and terminated:
            if self._PlanFlip(q_robot_full, t):
                state[0] = self.STATE_ATTEMPTING_FLIP
                print "State machine attempting flip"
            else:
                state[0] = self.STATE_STUCK
                self.q_traj = None
                self.q_traj_d = None
                print "State machine stuck"

	if state[0] == self.STATE_STUCK:
		raise StopIteration

    def _DoCalcKukaSetpointOutput(self, context, y_data):
        t = context.get_time()
        if self.q_traj:
            target_q = self.q_traj.value(t)[
                self.kuka_controller.controlled_inds, 0]
            target_v = self.q_traj_d.value(t)[
                self.kuka_controller.controlled_inds, 0]
            if t >= self.q_traj.end_time():
                target_v *= 0.
        else:
            x = self.EvalVectorInput(
                context, self.robot_state_input_port.get_index()).get_value()
            target_q = x[0:self.nq_kuka]
            target_v = target_q*0.

        kuka_setpoint = y_data.get_mutable_value()
        nq_plan = target_q.shape[0]
        kuka_setpoint[:nq_plan] = target_q[:]
        kuka_setpoint[self.nq_full:(self.nq_full+nq_plan)] = target_v[:]

    def _DoCalcHandSetpointOutput(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        #if self.q_traj:
        #    q_hand = self.q_traj.value(context.get_time())[
        #        self.hand_controller.controlled_inds, 0]
        #    y[:] = q_hand[1] - q_hand[0]
        #else:
        #    y[:] = 0.
        y[:] = 0.
