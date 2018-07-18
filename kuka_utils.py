# -*- coding: utf8 -*-

from copy import deepcopy
import os.path
from matplotlib import cm
import numpy as np
import time

import pydrake
from pydrake.all import (
    AddFlatTerrainToWorld,
    AddModelInstancesFromSdfString,
    AddModelInstanceFromUrdfFile,
    AddModelInstanceFromUrdfStringSearchingInRosPackages,
    FloatingBaseType,
    LeafSystem,
    PortDataType,
    RigidBodyFrame,
    RigidBodyTree,
    RollPitchYaw,
    RotationMatrix
)
from pydrake.solvers import ik

import kuka_ik

import meshcat
import meshcat.transformations as tf
import meshcat.geometry as g


def extract_position_indices(rbt, controlled_joint_names):
    ''' Given a RigidBodyTree and a list of
    joint names, returns, in separate lists, the
    position indices (i.e. offsets into the RBT positions vector)
    corresponding to those joints, and the rest of the
    position indices. '''
    controlled_config_inds = []
    other_config_inds = []
    for i in range(rbt.get_num_bodies()):
        body = rbt.get_body(i)
        if body.has_joint():
            joint = body.getJoint()
            if joint.get_name() in controlled_joint_names:
                controlled_config_inds += range(
                    body.get_position_start_index(),
                    body.get_position_start_index() +
                    joint.get_num_positions())
            else:
                other_config_inds += range(
                    body.get_position_start_index(),
                    body.get_position_start_index() +
                    joint.get_num_positions())
    if len(controlled_joint_names) != len(controlled_config_inds):
        raise ValueError("Didn't find all "
                         "requested controlled joint names.")

    return controlled_config_inds, other_config_inds


def is_trajectory_collision_free(rbt, qtraj, sample_time=0.1):
    print "is_trajectory_collision_free not implemented"
    return True
    for t in np.arange(qtraj.start_time(), qtraj.end_time(), sample_time):
        q = qtraj.value(t)
        kinsol = rbt.doKinematics(q)
        pointpairs = rbt.ComputeMaximumDepthCollisionPoints(kinsol)
        # do something with those
    return False


def visualize_plan_with_meshcat(rbt, pbrv, qtraj, sample_time=0.05):
    for t in np.arange(qtraj.start_time(), qtraj.end_time(), sample_time):
        q = qtraj.value(t)[:]
        pbrv.draw(q)
        time.sleep(sample_time)



class ExperimentWorldBuilder():
    def __init__(self):
        self.table_top_z_in_world = 0.736 + 0.057 / 2
        self.manipuland_body_indices = []
        self.manipuland_params = []
        self.tabletop_indices = []
        self.model_index_dict = {}

    def add_model_wrapper(self, filename, floating_base_type, frame, rbt):
        if filename.split(".")[-1] == "sdf":
            model_instance_map = AddModelInstancesFromSdfString(
                open(filename).read(), floating_base_type, frame, rbt)
        else:
            model_instance_map   = AddModelInstanceFromUrdfFile(
                filename, floating_base_type, frame, rbt)
        for key in model_instance_map.keys():
            self.model_index_dict[key] = model_instance_map[key]

    def setup_kuka(self, rbt):
        iiwa_urdf_path = os.path.join(
            pydrake.getDrakePath(),
            "manipulation", "models", "iiwa_description", "urdf",
            "iiwa14_polytope_collision.urdf")

        wsg50_sdf_path = os.path.join(
            pydrake.getDrakePath(),
            "manipulation", "models", "wsg_50_description", "sdf",
            "schunk_wsg_50.sdf")

        table_sdf_path = os.path.join(
            pydrake.getDrakePath(),
            "examples", "kuka_iiwa_arm", "models", "table",
            "extra_heavy_duty_table_surface_only_collision.sdf")

        AddFlatTerrainToWorld(rbt)
        table_frame_robot = RigidBodyFrame(
            "table_frame_robot", rbt.world(),
            [0.0, 0, 0], [0, 0, 0])
        self.add_model_wrapper(table_sdf_path, FloatingBaseType.kFixed,
            table_frame_robot, rbt)
        self.tabletop_indices.append(rbt.get_num_bodies()-1)
        table_frame_fwd = RigidBodyFrame(
            "table_frame_fwd", rbt.world(),
            [0.7, 0, 0], [0, 0, 0])
        self.add_model_wrapper(table_sdf_path, FloatingBaseType.kFixed,
            table_frame_fwd, rbt)
        self.tabletop_indices.append(rbt.get_num_bodies()-1)

        robot_base_frame = RigidBodyFrame(
            "robot_base_frame", rbt.world(),
            [0.0, 0, self.table_top_z_in_world], [0, 0, 0])
        self.add_model_wrapper(iiwa_urdf_path, FloatingBaseType.kFixed,
            robot_base_frame, rbt)
        
        # Add gripper
        gripper_frame = rbt.findFrame("iiwa_frame_ee")
        self.add_model_wrapper(wsg50_sdf_path, FloatingBaseType.kFixed,
            gripper_frame, rbt)


    def add_cut_cylinder_to_tabletop(self, rbt, model_name,
        do_convex_decomp=False, height=None, radius=None,
        cut_dirs=None, cut_points=None):
        import mesh_creation
        import trimesh
        # Determine parameters of the cylinders
        height = height or np.random.random() * 0.03 + 0.04
        radius = radius or np.random.random() * 0.02 + 0.01
        if cut_dirs is None:
            cut_dirs = [np.array([1., 0., 0.])]
        if cut_points is None:
            cut_points = [np.array([(np.random.random() - 0.5)*radius*1., 0, 0])]
        cutting_planes = zip(cut_points, cut_dirs)
        print "Cutting with cutting planes ", cutting_planes
        # Create a mesh programmatically for that cylinder
        cyl = mesh_creation.create_cut_cylinder(
            radius, height, cutting_planes, sections=10)
        cyl.density = 1000.  # Same as water
        
        self.manipuland_params.append(dict(
                height=height,
                radius=radius,
                cut_dirs=cut_dirs,
                cut_points=cut_points
            ))
        # Save it out to a file and add it to the RBT
        object_init_frame = RigidBodyFrame(
            "object_init_frame_%s" % model_name, rbt.world(),
            [0., 0., 0.], [0., 0., 0.])

        if do_convex_decomp:  # more powerful, does a convex decomp
            urdf_dir = "/tmp/mesh_%s/" % model_name
            trimesh.io.urdf.export_urdf(cyl, urdf_dir)
            urdf_path = urdf_dir + "mesh_%s.urdf" % model_name
            self.add_model_wrapper(urdf_path, FloatingBaseType.kRollPitchYaw,
                                   object_init_frame, rbt)
            self.manipuland_body_indices.append(rbt.get_num_bodies()-1)
        else:
            sdf_dir = "/tmp/mesh_%s/" % model_name
            file_name = "mesh_%s" % model_name
            mesh_creation.export_sdf(
                cyl, file_name, sdf_dir, color=[0.75, 0.5, 0.2, 1.])
            sdf_path = sdf_dir + "mesh_%s.sdf" % model_name
            self.add_model_wrapper(sdf_path, FloatingBaseType.kRollPitchYaw,
                                   object_init_frame, rbt)
            self.manipuland_body_indices.append(rbt.get_num_bodies()-1)

    def project_rbt_to_nearest_feasible_on_table(self, rbt, q0):
        # Project arrangement to nonpenetration with IK
        constraints = []

        constraints.append(ik.MinDistanceConstraint(
            model=rbt, min_distance=1E-3,
            active_bodies_idx=self.manipuland_body_indices + self.tabletop_indices,
            active_group_names=set()))

        locked_position_inds = []
        for body_i in range(rbt.get_num_bodies()):
            if body_i in self.manipuland_body_indices:
                constraints.append(ik.WorldPositionConstraint(
                    model=rbt, body=body_i,
                    pts=np.array([0., 0., 0.]),
                    lb=np.array([0.4, -0.2, self.table_top_z_in_world]),
                    ub=np.array([0.6, 0.2, self.table_top_z_in_world+0.3])))
            else:
                body = rbt.get_body(body_i)
                if body.has_joint():
                    for k in range(body.get_position_start_index(),
                                   body.get_position_start_index() +
                                   body.getJoint().get_num_positions()):
                        locked_position_inds.append(k)

        required_posture_constraint = ik.PostureConstraint(rbt)
        required_posture_constraint.setJointLimits(
            locked_position_inds, q0[locked_position_inds]-0.001,
            q0[locked_position_inds]+0.001)
        constraints.append(required_posture_constraint)

        options = ik.IKoptions(rbt)
        options.setMajorIterationsLimit(10000)
        options.setIterationsLimit(100000)
        results = ik.InverseKin(
            rbt, q0, q0, constraints, options)

        qf = results.q_sol[0]
        info = results.info[0]
        print "Projected to feasibility with info %d" % info
        return qf


def render_system_with_graphviz(system, output_file="system_view.gz"):
    ''' Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file. '''
    from graphviz import Source
    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)


class RgbdCameraMeshcatVisualizer(LeafSystem):
    def __init__(self,
                 camera,
                 rbt,
                 draw_timestep=0.033333,
                 prefix="RBCameraViz",
                 zmq_url="tcp://127.0.0.1:6000"):
        LeafSystem.__init__(self)
        self.set_name('camera meshcat visualization')
        self.timestep = draw_timestep
        self._DeclarePeriodicPublish(draw_timestep, 0.0)
        self.camera = camera
        self.rbt = rbt
        self.prefix = prefix

        self.camera_input_port = \
            self._DeclareInputPort(PortDataType.kAbstractValued,
                                   camera.depth_image_output_port().size())
        self.state_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   rbt.get_num_positions() +
                                   rbt.get_num_velocities())

        # Set up meshcat
        self.vis = meshcat.Visualizer(zmq_url=zmq_url)
        self.vis[prefix].delete()

    def _DoPublish(self, context, event):
        u_data = self.EvalAbstractInput(context, 0).get_value()
        x = self.EvalVectorInput(context, 1).get_value()
        w, h, _ = u_data.data.shape
        depth_image = u_data.data[:, :, 0]

        # Convert depth image to point cloud, with +z being
        # camera "forward"
        Kinv = np.linalg.inv(
            self.camera.depth_camera_info().intrinsic_matrix())
        U, V = np.meshgrid(np.arange(h), np.arange(w))
        points_in_camera_frame = np.vstack([
            U.flatten(),
            V.flatten(),
            np.ones(w*h)])
        points_in_camera_frame = Kinv.dot(points_in_camera_frame) * \
            depth_image.flatten()

        # The depth camera has some offset from the camera's root frame,
        # so take than into account.
        pose_mat = self.camera.depth_camera_optical_pose().matrix()
        points_in_camera_frame = pose_mat[0:3, 0:3].dot(points_in_camera_frame)
        points_in_camera_frame += np.tile(pose_mat[0:3, 3], [w*h, 1]).T

        kinsol = self.rbt.doKinematics(x[:self.rbt.get_num_positions()])
        points_in_world_frame = self.rbt.transformPoints(
            kinsol,
            points_in_camera_frame,
            self.camera.frame().get_frame_index(),
            0)

        # Color points according to their normalized height
        min_height = 0.6
        max_height = 0.9
        colors = cm.jet(
            (points_in_world_frame[2, :]-min_height)/(max_height-min_height)
            ).T[0:3, :]

        self.vis[self.prefix]["points"].set_object(
            g.PointCloud(position=points_in_world_frame,
                         color=colors,
                         size=0.005))
