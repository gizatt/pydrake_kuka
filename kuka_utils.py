# -*- coding: utf8 -*-

from copy import deepcopy
import os.path
from matplotlib import cm
import numpy as np
import subprocess
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

from underactuated import MeshcatRigidBodyVisualizer


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
    def __init__(self, with_knife=True):
        self.table_top_z_in_world = 0.736 + 0.057 / 2
        self.manipuland_body_indices = []
        self.manipuland_params = []
        self.tabletop_indices = []
        self.guillotine_blade_index = None
        self.with_knife = with_knife
        self.model_index_dict = {}
        r, p, y = 2.4, 1.9, 3.8
        self.magic_rpy_offset = np.array([r, p, y])
        self.magic_rpy_rotmat = RotationMatrix(
            RollPitchYaw(r, p, y)).matrix()

    def add_model_wrapper(self, filename, floating_base_type, frame, rbt):
        if filename.split(".")[-1] == "sdf":
            model_instance_map = AddModelInstancesFromSdfString(
                open(filename).read(), floating_base_type, frame, rbt)
        else:
            model_instance_map = AddModelInstanceFromUrdfFile(
                filename, floating_base_type, frame, rbt)
        for key in model_instance_map.keys():
            self.model_index_dict[key] = model_instance_map[key]

    def setup_initial_world(self, n_objects):
        # Construct the initial robot and its environment
        rbt = RigidBodyTree()
        self.setup_kuka(rbt)

        rbt_just_kuka = rbt.Clone()
        rbt_just_kuka.compile()
        # Figure out initial pose for the arm
        ee_body = rbt_just_kuka.FindBody("right_finger").get_body_index()
        ee_point = np.array([0.0, 0.03, 0.0])
        end_effector_desired = np.array(
            [0.5, 0.0, self.table_top_z_in_world+0.5, -np.pi/2., 0., 0.])
        q0_kuka_seed = rbt_just_kuka.getZeroConfiguration()
        # "Center low" from IIWA stored_poses.json from Spartan
        # + closed hand + raised blade
        q0_kuka_seed[0:9] = np.array([-0.18, -1., 0.12, -1.89, 0.1, 1.3, 0.38,
                                      0.0, 0.0])
        if self.with_knife:
            q0_kuka_seed[9] = 1.5

        q0_kuka, info = kuka_ik.plan_ee_configuration(
            rbt_just_kuka, q0_kuka_seed, q0_kuka_seed, end_effector_desired,
            ee_body, ee_point, allow_collision=True, euler_limits=0.01)
        if info != 1:
            print "Info %d on IK for initial posture." % info

        # Add objects + make random initial poses
        q0 = np.zeros(rbt.get_num_positions() + 6*n_objects)
        q0[0:rbt_just_kuka.get_num_positions()] = q0_kuka
        for k in range(n_objects):
            self.add_cut_cylinder_to_tabletop(rbt, "cyl_%d" % k,
                                              cut_dirs=[], cut_points=[])
            radius = self.manipuland_params[-1]["radius"]
            new_body = rbt.get_body(self.manipuland_body_indices[-1])

            # Remember to reverse effects of self.magic_rpy_offset
            new_pos = self.magic_rpy_rotmat.T.dot(np.array(
                        [0.4 + np.random.random()*0.2,
                         -0.2 + np.random.random()*0.4,
                         self.table_top_z_in_world+radius+0.001]))

            new_rot = (np.random.random(3) * np.pi * 2.) - \
                self.magic_rpy_offset
            q0[range(new_body.get_position_start_index(),
                     new_body.get_position_start_index()+6)] = np.hstack([
                        new_pos, new_rot])
        rbt.compile()
        q0_feas = self.project_rbt_to_nearest_feasible_on_table(
            rbt, q0)
        return rbt, rbt_just_kuka, q0_feas

    def do_cut(self, rbt, x, cut_body_index, cut_pt, cut_normal):
        # Rebuilds the full rigid body tree, replacing cut_body_index
        # with one that is cut, but otherwise keeping the rest of the
        # tree the same. The new tree won't have the same body indices
        # (for the manipulands and anything added after them) as the
        # original.
        old_manipuland_indices = deepcopy(self.manipuland_body_indices)
        old_manipuland_params = deepcopy(self.manipuland_params)
        self.__init__()

        new_rbt = RigidBodyTree()
        self.setup_kuka(new_rbt)

        q_new = np.zeros(rbt.get_num_positions() + 6)
        v_new = np.zeros(rbt.get_num_positions() + 6)
        if rbt.get_num_positions() != rbt.get_num_velocities():
            raise Exception("Can't handle nq != nv, sorry...")

        def copy_state(from_indices, to_indices):
            q_new[to_indices] = x[from_indices]
            v_new[to_indices] = x[np.array(from_indices) +
                                  rbt.get_num_positions()]

        copy_state(range(new_rbt.get_num_positions()),
                   range(new_rbt.get_num_positions()))

        k = 0
        for i, ind in enumerate(old_manipuland_indices):
            print i, ind
            p = old_manipuland_params[i]
            if ind is cut_body_index:
                for sign in [-1., 1.]:
                    try:
                        self.add_cut_cylinder_to_tabletop(
                            new_rbt, "cyl_%d" % k,
                            height=p["height"],
                            radius=p["radius"],
                            cut_dirs=p["cut_dirs"] + [cut_normal*sign],
                            cut_points=p["cut_points"] +
                            [cut_pt + cut_normal*sign*0.002])
                    except subprocess.CalledProcessError as e:
                        print "Failed a cut: ", e
                        continue  # failed to cut
                    k += 1
                    copy_state(
                        range(rbt.get_body(ind).get_position_start_index(),
                              rbt.get_body(ind).get_position_start_index()
                              + 6),
                        range(new_rbt.get_num_positions()-6,
                              new_rbt.get_num_positions()))
            else:
                self.add_cut_cylinder_to_tabletop(
                    new_rbt, "cyl_%d" % k,
                    height=p["height"],
                    radius=p["radius"],
                    cut_dirs=p["cut_dirs"],
                    cut_points=p["cut_points"])
                copy_state(range(rbt.get_body(ind).get_position_start_index(),
                                 rbt.get_body(ind).get_position_start_index()
                                 + 6),
                           range(new_rbt.get_num_positions()-6,
                                 new_rbt.get_num_positions()))
                k += 1

        # Account for possible cut failures
        q_new = q_new[:new_rbt.get_num_positions()]
        v_new = v_new[:new_rbt.get_num_velocities()]

        # Map old state into new state
        new_rbt.compile()
        q_new = self.project_rbt_to_nearest_feasible_on_table(new_rbt, q_new)
        return new_rbt, np.hstack([q_new, v_new])

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

        guillotine_path = "guillotine.sdf"

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

        if self.with_knife:
            # Add guillotine
            guillotine_frame = RigidBodyFrame(
                "guillotine_frame", rbt.world(),
                [0.6, -0.41, self.table_top_z_in_world], [0, 0, 0])
            self.add_model_wrapper(guillotine_path, FloatingBaseType.kFixed,
                                   guillotine_frame, rbt)
            self.guillotine_blade_index = \
                rbt.FindBody("blade").get_body_index()

    def add_cut_cylinder_to_tabletop(
            self, rbt, model_name, do_convex_decomp=False, height=None,
            radius=None, cut_dirs=None, cut_points=None):
        import mesh_creation
        import trimesh
        # Determine parameters of the cylinders
        height = height or np.random.random() * 0.05 + 0.1
        radius = radius or np.random.random() * 0.02 + 0.01
        if cut_dirs is None:
            cut_dirs = [np.array([1., 0., 0.])]
        if cut_points is None:
            cut_points = [np.array(
                [(np.random.random() - 0.5)*radius*1., 0, 0])]
        cutting_planes = zip(cut_points, cut_dirs)
        print "Cutting with cutting planes ", cutting_planes
        # Create a mesh programmatically for that cylinder
        cyl = mesh_creation.create_cut_cylinder(
            radius, height, cutting_planes, sections=6)
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
            np.zeros(3),
            self.magic_rpy_offset)

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

        q0 = np.clip(q0, rbt.joint_limit_min, rbt.joint_limit_max)

        constraints.append(ik.MinDistanceConstraint(
            model=rbt, min_distance=1E-3,
            active_bodies_idx=self.manipuland_body_indices +
            self.tabletop_indices,
            active_group_names=set()))

        options = ik.IKoptions(rbt)
        options.setMajorIterationsLimit(10000)
        options.setIterationsLimit(100000)
        options.setQ(np.eye(rbt.get_num_positions())*1E6)
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
