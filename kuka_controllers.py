# -*- coding: utf8 -*-
import functools
from copy import deepcopy
import math
import numpy as np
import random
import time

import pydrake
from pydrake.all import (
    AbstractValue,
    AutoDiffXd,
    BasicVector,
    LeafSystem,
    MathematicalProgram,
    PiecewisePolynomial,
    PortDataType,
)
from pydrake.forwarddiff import jacobian

import kuka_ik
import kuka_utils

from underactuated import MeshcatRigidBodyVisualizer


class InstantaneousKukaControllerSetpoint():
    def __init__(
        self, ee_pt_des=None, ee_v_des=None, ee_frame=None, 
        ee_pt=None, ee_x_des=None, ee_y_des=None, ee_z_des=None,
        q_des=None, v_des=None, Kq=0., Kv=0., Ka=0., Kee_pt=0.,
        Kee_xyz=0., Kee_xyzd=0., Kee_v=0.):
        self.ee_frame = ee_frame    # frame id
        self.ee_pt = ee_pt          # 3x1, pt in ee frame
        self.ee_pt_des = ee_pt_des    # 3x1, world frame
        self.ee_x_des = ee_x_des    # 3x1, world frame, where ee frame +x axis should point in world frame
        self.ee_y_des = ee_y_des    # 3x1, world frame, where ee frame +y axis should point in world frame
        self.ee_z_des = ee_z_des    # 3x1, world frame, where ee frame +z axis should point in world frame
        self.ee_v_des = ee_v_des    # 3x1, world frame
        self.q_des = q_des          # nq x 1
        self.v_des = v_des          # nv x 1
        self.Kq = Kq                # nq x nq or scalar
        self.Kv = Kv                # nv x nv or scalar
        self.Ka = Ka                # nv x nv or scalar
        self.Kee_pt = Kee_pt          # 3x3 or scalar
        self.Kee_xyz = Kee_xyz      # 3x3 or scalar
        self.Kee_xyzd = Kee_xyzd      # 3x3 or scalar
        self.Kee_v = Kee_v          # 3x3 or scalar

    def Copy(self, setpoint):
        for a in dir(self):
            if not a.startswith('__') and a is not "Copy":
                setattr(self, a, deepcopy(getattr(setpoint, a)))


class InstantaneousKukaController(LeafSystem):
    def __init__(self, rbt, plant,
                 control_period=0.033,
                 print_period=0.5):
        LeafSystem.__init__(self)
        self.set_name("Instantaneous Kuka Controller")

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
            raise RuntimeError("The joint set specified is underactuated.")

        self.B_inv = np.linalg.inv(self.B)

        # Copy lots of stuff
        self.rbt = rbt
        self.nq = rbt.get_num_positions()
        self.plant = plant
        self.nu = plant.get_input_port(0).size() # hopefully matches
        if self.nu != self.nq_reduced:
            raise RuntimeError("plant input port not equal to number of controlled joints")
        self.print_period = print_period
        self.last_print_time = -print_period
        self.shut_up = False
        self.control_period = control_period

        self.robot_state_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   rbt.get_num_positions() +
                                   rbt.get_num_velocities())

        self.setpoint_input_port = \
            self._DeclareInputPort(PortDataType.kAbstractValued, 0)

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
        setpoint = self.EvalAbstractInput(
            context, self.setpoint_input_port.get_index()).get_value()

        q_full = x[:self.nq]
        v_full = x[self.nq:]
        q = x[self.controlled_inds]
        v = x[self.nq:][self.controlled_inds]

        kinsol = self.rbt.doKinematics(q_full, v_full)

        M_full = self.rbt.massMatrix(kinsol)
        C = self.rbt.dynamicsBiasTerm(kinsol, {}, None)[self.controlled_inds]
        # Python slicing doesn't work in 2D, so do it row-by-row
        M = np.zeros((self.nq_reduced, self.nq_reduced))
        for k in range(self.nq_reduced):
            M[:, k] = M_full[self.controlled_inds, k]

        # Pick a qdd that results in minimum deviation from the desired
        # end effector pose (according to the end effector frame's jacobian
        # at the current state)
        # v_next = v + control_period * qdd
        # q_next = q + control_period * (v + v_next) / 2.
        # ee_v = J*v
        # ee_p = from forward kinematics
        # ee_v_next = J*v_next
        # ee_p_next = ee_p + control_period * (ee_v + ee_v_next) / 2.
        # min  u and qdd
        #        || q_next - q_des ||_Kq
        #     +  || v_next - v_des ||_Kv
        #     +  || qdd ||_Ka
        #     +  || Kee_v - ee_v_next ||_Kee_pt
        #     +  || Kee_pt - ee_p_next ||_Kee_v
        #     +  the messily-implemented angular ones?
        # s.t. M qdd + C = B u
        # (no contact modeling for now)
        prog = MathematicalProgram()
        qdd = prog.NewContinuousVariables(self.nq_reduced, "qdd")
        u = prog.NewContinuousVariables(self.nu, "u")

        prog.AddQuadraticCost(qdd.T.dot(setpoint.Ka).dot(qdd))

        v_next = v + self.control_period * qdd
        q_next = q + self.control_period * (v + v_next) / 2.
        if setpoint.v_des is not None:
            v_err = setpoint.v_des - v_next
            prog.AddQuadraticCost(v_err.T.dot(setpoint.Kv).dot(v_err))
        if setpoint.q_des is not None:
            q_err = setpoint.q_des - q_next
            prog.AddQuadraticCost(q_err.T.dot(setpoint.Kq).dot(q_err))

        if setpoint.ee_frame is not None and setpoint.ee_pt is not None:
            # Convert x to autodiff for Jacobians computation
            q_full_ad = np.empty(self.nq, dtype=AutoDiffXd)
            for i in range(self.nq):
                der = np.zeros(self.nq)
                der[i] = 1
                q_full_ad.flat[i] = AutoDiffXd(q_full.flat[i], der)
            kinsol_ad = self.rbt.doKinematics(q_full_ad)

            tf_ad = self.rbt.relativeTransform(kinsol_ad, 0, setpoint.ee_frame)
            
            # Compute errors in EE pt position and velocity (in world frame)
            ee_p_ad = tf_ad[0:3, 0:3].dot(setpoint.ee_pt) + tf_ad[0:3, 3]
            ee_p = np.hstack([y.value() for y in ee_p_ad])

            J_ee = np.vstack([y.derivatives() for y in ee_p_ad]).reshape(3, self.nq)
            J_ee = J_ee[:, self.controlled_inds]

            ee_v = J_ee.dot(v)
            ee_v_next = J_ee.dot(v_next)
            ee_p_next = ee_p + self.control_period * (ee_v + ee_v_next) / 2.

            if setpoint.ee_pt_des is not None:
                ee_p_err = setpoint.ee_pt_des.reshape((3, 1)) - ee_p_next.reshape((3, 1))
                prog.AddQuadraticCost((ee_p_err.T.dot(setpoint.Kee_pt).dot(ee_p_err))[0, 0])
            if setpoint.ee_v_des is not None:
                ee_v_err = setpoint.ee_v_des.reshape((3, 1)) - ee_v_next.reshape((3, 1))
                prog.AddQuadraticCost((ee_v_err.T.dot(setpoint.Kee_v).dot(ee_v_err))[0, 0])

            # Also compute errors in EE cardinal vector directions vs targets in world frame
            for i, vec in enumerate(
                    (setpoint.ee_x_des, setpoint.ee_y_des, setpoint.ee_z_des)):
                if vec is not None:
                    direction = np.zeros(3)
                    direction[i] = 1.
                    ee_dir_ad = tf_ad[0:3, 0:3].dot(direction)
                    ee_dir_p = np.hstack([y.value() for y in ee_dir_ad])
                    J_ee_dir = np.vstack([y.derivatives() for y in ee_dir_ad]).reshape(3, self.nq)
                    J_ee_dir = J_ee_dir[:, self.controlled_inds]
                    ee_dir_v = J_ee_dir.dot(v)
                    ee_dir_v_next = J_ee_dir.dot(v_next)
                    ee_dir_p_next = ee_dir_p + self.control_period * (ee_dir_v + ee_dir_v_next) / 2.
                    ee_dir_p_err = vec.reshape((3, 1)) - ee_dir_p_next.reshape((3, 1))
                    prog.AddQuadraticCost((ee_dir_p_err.T.dot(setpoint.Kee_xyz).dot(ee_dir_p_err))[0, 0])
                    prog.AddQuadraticCost((ee_dir_v_next.T.dot(setpoint.Kee_xyzd).dot(ee_dir_v_next)))



        LHS = np.dot(M, qdd) + C
        RHS = np.dot(self.B, u)
        for i in range(self.nq_reduced):
            prog.AddLinearConstraint(LHS[i] == RHS[i])

        result = prog.Solve()
        u = prog.GetSolution(u)
        new_control_input[:] = u

    def _DoCalcVectorOutput(self, context, y_data):
        if (self.print_period and
                context.get_time() - self.last_print_time
                >= self.print_period):
            print "t: ", context.get_time()
            self.last_print_time = context.get_time()
        control_output = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
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

        self.max_force = 50.  # gripper max closing / opening force

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


class GuillotineController(LeafSystem):
    def __init__(self, rbt, plant,
                 control_period=0.001):
        LeafSystem.__init__(self)
        self.set_name("GuillotineController Controller")

        self.controlled_joint_names = [
            "knife_joint"
        ]

        self.max_force = 100.

        self.controlled_inds, _ = kuka_utils.extract_position_indices(
            rbt, self.controlled_joint_names)

        self.nu = plant.get_input_port(2).size()
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

        q_des = self.EvalVectorInput(
            context, self.setpoint_input_port.get_index()).get_value()

        q_full = x[:self.nq]
        v_full = x[self.nq:]

        q = q_full[self.controlled_inds]
        v = v_full[self.controlled_inds]
        v_des = np.zeros(1)

        qerr = q_des - q
        verr = v_des - v

        Kp = 1000.
        Kv = 100.
        new_control_input[:] = np.clip(
            Kp * qerr + Kv * verr, -self.max_force, self.max_force)

    def _DoCalcVectorOutput(self, context, y_data):
        control_output = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        y[:] = control_output[:]


class TaskPrimitiveContextInfo():
    def __init__(self, t, x):
        self.t = t
        self.x = x

class TaskPrimitive():
    def __init__(self):
        ''' Functions *must* takes ContextInfo, setpoint_object as their args. '''
        self.functions = {}
        self.guards = {}
        self.current_function_name = ""

    def CalcSetpointsOutput(self, context_info, setpoint_object,
                            gripper_setpoint, knife_setpoint):
        if self.current_function_name == "":
            raise RuntimeError("TaskPrimitive initial function not initialized.")

        # Check guards
        for guard_handle, to_name in self.guards[self.current_function_name]:
            if guard_handle(context_info):
                self.current_function_name = to_name

        # Run function
        self.functions[self.current_function_name](
            context_info, setpoint_object, gripper_setpoint, knife_setpoint)

    @staticmethod
    def CalcExpectedCost(self, context_info):
        ''' Can return Infinity if using this primitive,
            given the context info, is infeasible or pointless. '''
        raise NotImplementedError()

    def _RegisterFunction(self, name, handle):
        self.functions[name] = handle
        self.guards[name] = []

    def _RegisterTransition(self, from_name, to_name, guard_handle):
        self.guards[from_name].append((guard_handle, to_name))


def MakeKukaNominalPoseSetpoint(rbt, q_nom):
    setpoint_object = InstantaneousKukaControllerSetpoint()
    setpoint_object.Ka = 1.0
    setpoint_object.Kq = 1000000.
    setpoint_object.Kv = 10000.

    setpoint_object.q_des = q_nom[0:7]
    setpoint_object.v_des = np.zeros(7)
    return setpoint_object

def RunNominalPoseTarget(context_info, setpoint_object,
                         gripper_setpoint, knife_setpoint,
                         template_setpoint):
    setpoint_object.Copy(template_setpoint)
    gripper_setpoint[:] = 0.5
    knife_setpoint[:] = np.pi/2.

class IdlePrimitive(TaskPrimitive):
    def __init__(self, rbt, q_nom):
        TaskPrimitive.__init__(self)

        self.current_function_name = "move to idle"
        self._RegisterFunction(
            "move to idle",
            functools.partial(RunNominalPoseTarget,
                              template_setpoint=
                              MakeKukaNominalPoseSetpoint(rbt, q_nom)))

    @staticmethod
    def CalcExpectedCost(self, context_info, rbt):
        return 1.


class CutPrimitive(TaskPrimitive):
    def __init__(self, rbt, q_nom):
        TaskPrimitive.__init__(self)
        self.rbt = rbt
        self.current_function_name = "clear hand from blade"
        self._RegisterFunction(
            "clear hand from blade",
            self.MoveIdle)
        self._RegisterTransition("clear hand from blade", "cut while moving to idle",
                                 self.IsHandClearOfBlade)
        self._RegisterFunction(
            "cut while moving to idle",
            self.MoveIdleAndChop)
        self._RegisterTransition("cut while moving to idle", "done",
                                 self.DoYouHearThePeopleSingle)

        self._RegisterFunction(
            "done",
            self.MoveIdle)

        self.ee_frame = self.rbt.findFrame(
            "iiwa_frame_ee").get_frame_index()
        self.base_setpoint = MakeKukaNominalPoseSetpoint(rbt, q_nom)

    def MoveIdle(self, context_info, setpoint_object,
                 gripper_setpoint, knife_setpoint):
        setpoint_object.Copy(self.base_setpoint)
        gripper_setpoint[:] = 0.5
        knife_setpoint[:] = np.pi/2.

    def MoveIdleAndChop(self, context_info, setpoint_object,
                 gripper_setpoint, knife_setpoint):
        setpoint_object.Copy(self.base_setpoint)
        gripper_setpoint[:] = 0.5
        knife_setpoint[:] = 0.

    def IsHandClearOfBlade(self, context_info):
        kinsol = self.rbt.doKinematics(context_info.x[:self.rbt.get_num_positions()])
        pt = self.rbt.transformPoints(kinsol, np.zeros(3), self.ee_frame, 0)
        if pt[1] >= -0.05 or pt[2] >= 1.2:
            return True

    def DoYouHearThePeopleSingle(self, context_info):
        # that is, has the guillotine blade fallen
        return context_info.x[9] < 0.05

    @staticmethod
    def CalcExpectedCost(self, context_info, rbt):
        return 1.


class MoveObjectPrimitive(TaskPrimitive):
    def __init__(self, rbt, q_nom, target_object_id, target_location):
        TaskPrimitive.__init__(self)

        self.rbt = rbt
        self.q_nom = q_nom
        self.target_object_id = target_object_id
        self.ee_frame = self.rbt.findFrame(
            "iiwa_frame_ee").get_frame_index()
        self.current_function_name = "move over object"
        self._RegisterFunction(
            "move over object",
            self.RunMoveOverObject)
        self._RegisterTransition("move over object", "move down to object",
                                 self.IsGripperOverObject)

        self._RegisterFunction(
            "move down to object",
            self.RunMoveToObject)
        self._RegisterTransition("move down to object", "grasp object",
                                 self.IsObjectInsideGripper)
        self._RegisterTransition("move down to object", "move over object",
                                 self.IsGripperNotOverObject)

        self._RegisterFunction(
            "grasp object",
            self.RunGraspObject)
        self._RegisterTransition("grasp object", "move down to object",
                                 self.IsObjectNotInsideGripper)
        self._RegisterTransition("grasp object", "move object",
                                 self.IsObjectGripped)

        self._RegisterFunction(
            "move object",
            self.RunMoveWithObject)
        self._RegisterTransition("move object", "move down to object",
                                 self.IsObjectNotInsideGripper)
        self._RegisterTransition("move object", "move down to object",
                                 self.IsObjectNotGripped)

        self.target_location = target_location
        self.object_pt = self.rbt.get_body(self.target_object_id).get_center_of_mass()

        self.base_setpoint = InstantaneousKukaControllerSetpoint()
        self.base_setpoint.Ka = 0.001
        self.base_setpoint.Kq = 0. # very weak, just regularizing
        self.base_setpoint.Kv = 10
        self.base_setpoint.Kee_v = 2000
        self.base_setpoint.Kee_xyzd = 2000.
        self.base_setpoint.q_des = self.q_nom[0:7]
        self.base_setpoint.v_des = np.zeros(7)
        self.base_setpoint.ee_frame = self.ee_frame
        self.base_setpoint.ee_pt = np.array([0., 0.045, 0.00])
        self.base_setpoint.ee_v_des = np.array([0., 0., 0.0])

    def RunMoveOverObject(self, context_info, setpoint_object,
                         gripper_setpoint, knife_setpoint):
        setpoint_object.Copy(self.base_setpoint)
        setpoint_object.Kee_pt = 1000000.
        setpoint_object.Kee_xyz = 500000.
        
        kinsol = self.rbt.doKinematics(context_info.x[:self.rbt.get_num_positions()])
        object_centroid_world = self.rbt.transformPoints(
            kinsol, self.object_pt, self.target_object_id, 0)
        object_px_world = self.rbt.transformPoints(
            kinsol, self.object_pt + np.array([0., 0., 1.]), self.target_object_id, 0)
        setpoint_object.ee_pt_des = object_centroid_world + np.array([[0., 0., 0.1]]).T
        y = context_info.x[self.rbt.get_body(self.target_object_id)
                                    .get_position_start_index()+5]
        setpoint_object.ee_z_des = object_px_world - object_centroid_world
        setpoint_object.ee_y_des = np.array([0., 0., -1.]) # facing down

        gripper_setpoint[:] = 0.5
        knife_setpoint[:] = np.pi/2.

    def RunMoveToObject(self, context_info, setpoint_object,
                        gripper_setpoint, knife_setpoint):
        setpoint_object.Copy(self.base_setpoint)
        setpoint_object.Kee_pt = 1000000.
        setpoint_object.Kee_xyz = 500000.
        kinsol = self.rbt.doKinematics(context_info.x[:self.rbt.get_num_positions()])
        object_centroid_world = self.rbt.transformPoints(
            kinsol, self.object_pt, self.target_object_id, 0)
        object_px_world = self.rbt.transformPoints(
            kinsol, self.object_pt + np.array([0., 0., 1.]), self.target_object_id, 0)
        setpoint_object.ee_pt_des = object_centroid_world + np.array([[0., 0., 0.]]).T
        y = context_info.x[self.rbt.get_body(self.target_object_id)
                                    .get_position_start_index()+5]
        setpoint_object.ee_z_des = object_px_world - object_centroid_world
        setpoint_object.ee_y_des = np.array([0., 0., -1.]) # facing down

        gripper_setpoint[:] = 0.5
        knife_setpoint[:] = np.pi/2.
        self.started_gripping_time = None # Please break this into a state or react to gripper force...

    def RunGraspObject(self, context_info, setpoint_object,
                        gripper_setpoint, knife_setpoint):
        setpoint_object.Copy(self.base_setpoint)
        setpoint_object.Kee_pt = 1000000.
        setpoint_object.Kee_xyz = 500000.
        kinsol = self.rbt.doKinematics(context_info.x[:self.rbt.get_num_positions()])
        object_centroid_world = self.rbt.transformPoints(
            kinsol, self.object_pt, self.target_object_id, 0)
        object_px_world = self.rbt.transformPoints(
            kinsol, self.object_pt + np.array([0., 0., 1.]), self.target_object_id, 0)
        setpoint_object.ee_pt_des = object_centroid_world + np.array([[0., 0., 0.]]).T
        y = context_info.x[self.rbt.get_body(self.target_object_id)
                                    .get_position_start_index()+5]
        setpoint_object.ee_z_des = object_px_world - object_centroid_world
        setpoint_object.ee_y_des = np.array([0., 0., -1.]) # facing down

        knife_setpoint[:] = np.pi/2.
        gripper_setpoint[:] = 0.0
        if self.started_gripping_time == None:
            self.started_gripping_time = context_info.t

    def RunMoveWithObject(self, context_info, setpoint_object,
                          gripper_setpoint, knife_setpoint):
        setpoint_object.Copy(self.base_setpoint)
        setpoint_object.Kee_pt = 1000000.
        setpoint_object.Kee_xyz = 500000.

        kinsol = self.rbt.doKinematics(context_info.x[:self.rbt.get_num_positions()])
        object_centroid_world = self.rbt.transformPoints(
            kinsol, self.object_pt, self.target_object_id, 0)
        # why is 3x1 + 3, a 3x3??????
        err = self.target_location.reshape((3, 1)) - object_centroid_world.reshape((3, 1))
        curr_ee_location = self.rbt.transformPoints(
            kinsol, self.base_setpoint.ee_pt, self.ee_frame, 0)
        offset = np.zeros((3, 1))
        if np.linalg.norm(err) > 0.2:
            offset[2] = 0.2 # up and over, I should really separate this into a new state
            if curr_ee_location[2] < 0.825:
                offset[0:2] -= err[0:2]
        setpoint_object.ee_pt_des = curr_ee_location.reshape((3, 1)) + err.reshape((3, 1)) + offset
        setpoint_object.ee_z_des = np.array([0., 1., 0.])
        setpoint_object.ee_y_des = np.array([0., 0., -1.]) # facing down

        gripper_setpoint[:] = 0.0
        knife_setpoint[:] = np.pi/2.

    def IsGripperOverObject(self, context_info):
        kinsol = self.rbt.doKinematics(context_info.x[:self.rbt.get_num_positions()])
        pt_ee = self.rbt.transformPoints(kinsol, self.base_setpoint.ee_pt, self.ee_frame, 0)
        pt_obj = self.rbt.transformPoints(kinsol, self.object_pt, self.target_object_id, 0)
        translation = pt_ee - pt_obj
        return np.linalg.norm(translation[0:2]) <= 0.1 and translation[2] >= 0.1

    def IsGripperNotOverObject(self, context_info):
        kinsol = self.rbt.doKinematics(context_info.x[:self.rbt.get_num_positions()])
        pt_ee = self.rbt.transformPoints(kinsol, self.base_setpoint.ee_pt, self.ee_frame, 0)
        pt_obj = self.rbt.transformPoints(kinsol, self.object_pt, self.target_object_id, 0)
        translation = pt_ee - pt_obj
        return np.linalg.norm(translation[0:2]) >= 0.1

    def IsObjectInsideGripper(self, context_info):
        kinsol = self.rbt.doKinematics(context_info.x[:self.rbt.get_num_positions()])
        pt_ee = self.rbt.transformPoints(kinsol, self.base_setpoint.ee_pt, self.ee_frame, 0)
        pt_obj = self.rbt.transformPoints(kinsol, self.object_pt, self.target_object_id, 0)
        translation = pt_ee - pt_obj
        return np.linalg.norm(translation) < 0.03

    def IsObjectNotInsideGripper(self, context_info):
        return not self.IsObjectInsideGripper(context_info)

    def IsObjectGripped(self, context_info):
        if self.IsObjectNotInsideGripper(context_info):
            return False
        if self.started_gripping_time is None or \
            (context_info.t - self.started_gripping_time) < 0.25:
            return False
        # Check finger closed state -- should be
        # mostly closed but not complete closed
        # TODO(gizatt) switch to gripper force
        finger_state = context_info.x[7:9]
        finger_closedness = finger_state[1] - finger_state[0]
        finger_speed = context_info.x[self.rbt.get_num_positions() + 8] - \
                       context_info.x[self.rbt.get_num_positions() + 7]
        if finger_closedness < 0.5 and finger_closedness > 0.01:
            return True
        else:
            return False

    def IsObjectNotGripped(self, context_info):
        return not self.IsObjectGripped(context_info)

    @staticmethod
    def CalcExpectedCost(context_info, rbt):
        return 1.

class TaskPlanner(LeafSystem):

    STATE_CLEARING_KNIFE_AREA = 1
    STATE_CUTTING_OBJECT = 2

    def __init__(self, rbt_full, q_nom, world_builder):
        LeafSystem.__init__(self)
        self.set_name("Task Planner")

        self.rbt = rbt_full
        self.world_builder = world_builder
        self.q_nom = np.array([-0.18, -1., 0.12, -1.89, 0.1, 1.3, 0.38, 0.0, 0.0, 1.5])
        self.nq_full = rbt_full.get_num_positions()
        self.nv_full = rbt_full.get_num_velocities()
        self.nu_full = rbt_full.get_num_actuators()

        self.robot_state_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   self.nq_full + self.nv_full)

        self._DeclareDiscreteState(1)
        self._DeclarePeriodicDiscreteUpdate(period_sec=0.1)
        # TODO set default state somehow better. Requires new
        # bindings to override AllocateDiscreteState or something else.
        self.initialized = False
        self.current_primitive = IdlePrimitive(self.rbt, q_nom)
        self.kuka_setpoint = self._DoAllocKukaSetpointOutput()
        # Put these in arrays so we can more easily pass by reference into
        # CalcSetpointsOutput
        self.gripper_setpoint = np.array([0.])
        self.knife_setpoint = np.array([np.pi/2.]) 

        # TODO The controller takes full state in, even though it only
        # controls the Kuka... that should be tidied up.
        self.kuka_setpoint_output_port = \
            self._DeclareAbstractOutputPort(
                self._DoAllocKukaSetpointOutput,
                self._DoCalcKukaSetpointOutput)

        self.hand_setpoint_output_port = \
            self._DeclareVectorOutputPort(BasicVector(1),
                                          self._DoCalcHandSetpointOutput)
        self.knife_setpoint_output_port = \
            self._DeclareVectorOutputPort(BasicVector(1),
                                          self._DoCalcKnifeSetpointOutput)

        self._DeclarePeriodicPublish(0.01, 0.0)

        # Really stupid simple state
        self.current_target_object = None
        self.current_target_object_move_location = None
        self.do_cut_after_current_move = False

    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        LeafSystem._DoCalcDiscreteVariableUpdates(
            self, context, events, discrete_state)

        state = discrete_state. \
            get_mutable_vector().get_mutable_value()
        if not self.initialized:
            state[0] = self.STATE_CLEARING_KNIFE_AREA
            self.initialized = True

        t = context.get_time()
        x_robot_full = self.EvalVectorInput(
            context, self.robot_state_input_port.get_index()).get_value()

        kinsol = self.rbt.doKinematics(x_robot_full[0:self.rbt.get_num_positions()])

        # Legendary spaghetti starts here
        if (isinstance(self.current_primitive, CutPrimitive) and
            self.current_primitive.current_function_name != "done"):
            # Currently cutting
            pass
        elif self.current_target_object is None:
            # Search over objects to find one to clear
            best_clear_object = None
            best_clear_dist = 100000.
            objects_on_table = []
            n_clear_objects = 0
            end_effector_pos = self.rbt.transformPoints(
                kinsol, np.zeros(3), 
                self.rbt.findFrame("iiwa_frame_ee").get_frame_index(), 0)
            for body_i in self.world_builder.manipuland_body_indices:
                current_object_pos = self.rbt.transformPoints(
                    kinsol, self.rbt.get_body(body_i).get_center_of_mass(), body_i, 0)
                if np.all(current_object_pos.T >= np.array([0.4, -0.6, 0.6])) and \
                   np.all(current_object_pos.T <= np.array([0.9, 0.6, 0.9])):
                    objects_on_table.append(body_i)
                if np.all(current_object_pos.T >= np.array([0.4, -0.6, 0.6])) and \
                   np.all(current_object_pos.T <= np.array([0.7, 0.0, 0.9])):
                    dist = np.linalg.norm(end_effector_pos - current_object_pos)
                    n_clear_objects += 1
                    if dist < best_clear_dist:
                        best_clear_dist = dist
                        best_clear_object = body_i

            if n_clear_objects > 1:  # If exactly 1 object, cut it!
                print "CLEARING OBJECT %d" % best_clear_object
                self.current_target_object_move_location = np.array([0.5+np.random.random()*0.2, 0.2, 0.825])
                self.current_target_object = best_clear_object
                self.do_cut_after_current_move = False
            elif n_clear_objects == 1:
                self.current_target_object_move_location = np.array([0.6, -0.2, 0.775])
                self.current_target_object = best_clear_object
                print "MOVING OBJECT %d FOR CUT" % self.current_target_object
                self.do_cut_after_current_move = True
            else:
                # Instead pick a random object that's on the table
                self.current_target_object_move_location = np.array([0.6, -0.2, 0.775])
                self.current_target_object = random.choice(objects_on_table)
                print "MOVING OBJECT %d FOR CUT" % self.current_target_object
                self.do_cut_after_current_move = True

            self.current_primitive = MoveObjectPrimitive(
                self.rbt, self.q_nom, 
                self.current_target_object, self.current_target_object_move_location)
        else:
            # Check that object location against current move goal
            current_object_pos = self.rbt.transformPoints(
                kinsol, self.rbt.get_body(self.current_target_object).get_center_of_mass(),
                self.current_target_object, 0)
            # Bail if we're really really lost
            if np.linalg.norm(current_object_pos.T - self.current_target_object_move_location) > 2.0 \
               or current_object_pos[2] < 0.5:
                print "BAILING"
                self.current_target_object = None
                self.current_target_object_move_location = None
                self.current_primitive = IdlePrimitive(self.rbt, self.q_nom)
                self.do_cut_after_current_move = False
            # Don't mode switch if we're currently moving super fast
            if np.max(np.abs(x_robot_full[self.nq_full:][0:7])) >= 0.25:
                # TODO switch this to
                pass
            elif self.do_cut_after_current_move:
                # Check that the COM of the object is close to the blade line
                if np.abs(current_object_pos[0] - 0.6) < 0.02 and current_object_pos[1] < -0.1:
                    print "EXECUTING CUT"
                    self.current_target_object = None
                    self.current_target_object_move_location = None
                    self.current_primitive = CutPrimitive(self.rbt, self.q_nom)
                    self.do_cut_after_current_move = False
            else:
                # Check that its' out of the blade area
                if np.any(current_object_pos.T <= np.array([0.4, -0.6, 0.6])) or \
                   np.any(current_object_pos.T >= np.array([0.7, 0.0, 0.9])):
                    print "EXECUTING IDLE AFTER CLEAR"
                    self.current_target_object = None
                    self.current_target_object_move_location = None
                    self.current_primitive = IdlePrimitive(self.rbt, self.q_nom)
                
        context_info = TaskPrimitiveContextInfo(t, x_robot_full)
        self.current_primitive.CalcSetpointsOutput(
            context_info, self.kuka_setpoint.get_mutable_value(),
            self.gripper_setpoint, self.knife_setpoint)

    def _DoAllocKukaSetpointOutput(self):
        return AbstractValue.Make(InstantaneousKukaControllerSetpoint())

    def _DoCalcKukaSetpointOutput(self, context, y_data):
        y_data.SetFrom(self.kuka_setpoint)

    def _DoCalcHandSetpointOutput(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y_data.get_mutable_value()[:] = self.gripper_setpoint

    def _DoCalcKnifeSetpointOutput(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y_data.get_mutable_value()[:] = self.knife_setpoint
