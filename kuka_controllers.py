# -*- coding: utf8 -*-
import functools
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
        Kee_xyz=0., Kee_v=0.):
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
        self.Kee_v = Kee_v          # 3x3 or scalar

    def Copy(self, setpoint):
        for a in dir(self):
            if not a.startswith('__'):
                setattr(self, a, getattr(setpoint, a))


class InstantaneousKukaController(LeafSystem):
    def __init__(self, rbt, plant,
                 control_period=0.05,
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
    setpoint_object.Kq = 10000.
    setpoint_object.Kv = 1000.

    setpoint_object.q_des = q_nom[0:7]
    setpoint_object.v_des = np.zeros(7)
    return setpoint_object

def RunNominalPoseTarget(context_info, setpoint_object,
                         gripper_setpoint, knife_setpoint,
                         template_setpoint):
    setpoint_object.Copy(template_setpoint)
    gripper_setpoint[:] = 0.
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


class GrabObjectPrimitive(TaskPrimitive):
    def __init__(self, rbt, q_nom, target_object_id):
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


        self.base_setpoint = InstantaneousKukaControllerSetpoint()
        self.base_setpoint.Ka = 1.0
        self.base_setpoint.Kq = 1000. # very weak, just regularizing
        self.base_setpoint.Kv = 1000.
        self.base_setpoint.Kee_v = 5000.
        self.base_setpoint.q_des = self.q_nom[0:7]
        self.base_setpoint.v_des = np.zeros(7)
        self.base_setpoint.ee_frame = self.ee_frame
        self.base_setpoint.ee_pt = np.array([0., 0.03, 0.])
        self.base_setpoint.ee_v_des = np.array([0., 0., 0.0])

    def RunMoveOverObject(self, context_info, setpoint_object,
                         gripper_setpoint, knife_setpoint):
        setpoint_object.Copy(self.base_setpoint)
        setpoint_object.Kee_pt = 1000000.
        setpoint_object.Kee_xyz = 500000.
        
        kinsol = self.rbt.doKinematics(context_info.x[:self.rbt.get_num_positions()])
        object_centroid_world = self.rbt.transformPoints(
            kinsol, np.zeros(3), self.target_object_id, 0)
        setpoint_object.ee_pt_des = object_centroid_world + np.array([[0., 0., 0.2]]).T
        setpoint_object.ee_x_des = np.array([1., 0., 0.])
        setpoint_object.ee_y_des = np.array([0., -1, -1.]) # facing down

        gripper_setpoint[:] = 0.5
        knife_setpoint[:] = np.pi/2.

    def RunMoveToObject(self, context_info, setpoint_object,
                        gripper_setpoint, knife_setpoint):
        setpoint_object.Copy(self.base_setpoint)
        setpoint_object.Kee_pt = 1000000.
        setpoint_object.Kee_xyz = 500000.
        
        kinsol = self.rbt.doKinematics(context_info.x[:self.rbt.get_num_positions()])
        object_centroid_world = self.rbt.transformPoints(
            kinsol, np.zeros(3), self.target_object_id, 0)
        setpoint_object.ee_pt_des = object_centroid_world + np.array([[0., 0., 0.0]]).T
        setpoint_object.ee_x_des = np.array([1., 0., 0.])
        setpoint_object.ee_y_des = np.array([0., -1, -1.]) # facing down

        gripper_setpoint[:] = 0.5
        knife_setpoint[:] = np.pi/2.

    def RunGraspObject(self, context_info, setpoint_object,
                        gripper_setpoint, knife_setpoint):
        self.RunMoveToObject(context_info, setpoint_object, gripper_setpoint, knife_setpoint)
        gripper_setpoint[:] = 0.0

    def IsGripperOverObject(self, context_info):
        kinsol = self.rbt.doKinematics(context_info.x[:self.rbt.get_num_positions()])
        tf = self.rbt.relativeTransform(kinsol, self.target_object_id, self.ee_frame)
        translation = tf[0:3, 3]
        print "current TF: ", translation, np.linalg.norm(translation)
        return np.linalg.norm(translation[0:2]) <= 0.2 and translation[2] >= 0.05

    def IsGripperNotOverObject(self, context_info):
        kinsol = self.rbt.doKinematics(context_info.x[:self.rbt.get_num_positions()])
        tf = self.rbt.relativeTransform(kinsol, self.target_object_id, self.ee_frame)
        translation = tf[0:3, 3]
        return np.linalg.norm(translation[0:2]) >= 0.2

    def IsObjectInsideGripper(self, context_info):
        kinsol = self.rbt.doKinematics(context_info.x[:self.rbt.get_num_positions()])
        tf = self.rbt.relativeTransform(kinsol, self.target_object_id, self.ee_frame)
        translation = tf[0:3, 3]
        print "current TF: ", translation, np.linalg.norm(translation)
        return np.linalg.norm(translation) < 0.05

    def IsObjectNotInsideGripper(self, context_info):
        return not self.IsObjectInsideGripper(context_info)

    @staticmethod
    def CalcExpectedCost(context_info, rbt):
        return 1.

class TaskPlanner(LeafSystem):

    STATE_STARTUP = -1

    def __init__(self, rbt_full, q_nom):
        LeafSystem.__init__(self)
        self.set_name("Task Planner")

        self.rbt_full = rbt_full
        self.q_nom = q_nom
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
        self.current_primitive = GrabObjectPrimitive(self.rbt_full, q_nom, self.rbt_full.FindBody("mesh_cyl_0").get_body_index())
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

    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        LeafSystem._DoCalcDiscreteVariableUpdates(
            self, context, events, discrete_state)

        state = discrete_state. \
            get_mutable_vector().get_mutable_value()
        if not self.initialized:
            state[0] = self.STATE_STARTUP
            self.initialized = True

        t = context.get_time()
        x_robot_full = self.EvalVectorInput(
            context, self.robot_state_input_port.get_index()).get_value()

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


class ManipTaskPlanner(LeafSystem):
    STATE_STARTUP = -1
    STATE_TERMINATE = -2
    STATE_MOVING_TO_NOMINAL = 0

    def __init__(self, rbt_full, rbt_kuka, q_nom,
                 world_builder, hand_controller,
                 kuka_controller, mrbv):
        LeafSystem.__init__(self)
        self.set_name("Food Flipping Task Planner")

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
                self.rbt_full, q0, self.q_nom, 10, 1.0, start_time)
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
                for t in np.linspace(-0.2, 0.2, 3):
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
        collision_inds.append(self.rbt_full.FindBody(
            "base", model_name="guillotine").get_body_index())
        collision_inds.append(self.rbt_full.FindBody(
            "blade", model_name="guillotine").get_body_index())
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
            ts = np.array([0., 0.5, 1.0, 1.5])
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
            state[0] = self.STATE_STARTUP
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
            state[0] = self.STATE_STARTUP
            #raise StopIteration

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
