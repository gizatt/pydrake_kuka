# -*- coding: utf8 -*-
import os
import random
import time

import cv2
import imageio
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import pydrake
import pydrake.math as drakemath
from pydrake.all import (
    AbstractValue,
    BasicVector,
    Image,
    LeafSystem,
    PixelType,
    PortDataType
)

class DepthImageCorruptionBlock(LeafSystem):
    def __init__(self, camera, save_dir):
        LeafSystem.__init__(self)
        self.set_name('depth image corruption superclass')
        self.camera = camera

        self.save_dir = save_dir
        if len(self.save_dir) > 0:
            os.system("rm -r %s" % self.save_dir)
            os.system("mkdir -p %s" % self.save_dir)

        self.depth_image_input_port = \
            self._DeclareInputPort(PortDataType.kAbstractValued,
                                   camera.depth_image_output_port().size())

        self.color_image_input_port = \
            self._DeclareInputPort(PortDataType.kAbstractValued,
                                   camera.color_image_output_port().size())

        self.depth_image_output_port = \
            self._DeclareAbstractOutputPort(
                self._DoAllocDepthCameraImage,
                self._DoCalcAbstractOutput)

    def _DoAllocDepthCameraImage(self):
        test = AbstractValue.Make(Image[PixelType.kDepth32F](
            self.camera.depth_camera_info().width(),
            self.camera.depth_camera_info().height()))
        return test

    def _DoCalcAbstractOutput(self, context, y_data):
        print "OVERRIDE ME"
        sys.exit(-1)


class DepthImageHeuristicCorruptionBlock(DepthImageCorruptionBlock):
    def __init__(self, camera, save_dir):
        DepthImageCorruptionBlock.__init__(self, camera, save_dir)
        self.set_name('depth image corruption, heuristic')

        self.rgbd_normal_limit = 0.5
        self.rgbd_noise = 0.001
        self.rgbd_projector_baseline = 0.      # 0.05
        self.rgbd_rectification_baseline = 0.  # -0.025
        self.near_distance = 0.2
        self.far_distance = 3.5

        # Cache these things that are used in every loop
        # to minimize re-allocation of these big arrays
        K = self.camera.depth_camera_info().intrinsic_matrix()
        K_rgb = self.camera.color_camera_info().intrinsic_matrix()
        w = self.camera.depth_camera_info().width()
        h = self.camera.depth_camera_info().height()
        # How much does each depth point project laterally
        # (in the axis of the camera-projector pair?)
        x_inds, y_inds = np.meshgrid(np.arange(w), np.arange(h))
        self.xy1_indices_im = np.dstack([
            x_inds, y_inds, np.ones((h, w))])
        self.iter = 0

    def _DoCalcAbstractOutput(self, context, y_data):
        start_time = time.time()

        u_data = self.EvalAbstractInput(context, 1).get_value()
        h, w, _ = u_data.data.shape
        rgb_image = np.empty((h, w), dtype=np.float64)
        rgb_image[:, :] = u_data.data[:, :, 0]

        if len(self.save_dir) > 0:
            save_image_uint8(
                "%s/%05d_rgb.png" % (self.save_dir, self.iter), rgb_image)

        u_data = self.EvalAbstractInput(context, 0).get_value()
        h, w, _ = u_data.data.shape
        depth_image = np.empty((h, w), dtype=np.float32)
        depth_image[:, :] = u_data.data[:, :, 0]
        good_mask = np.isfinite(depth_image)
        depth_image = np.clip(depth_image, self.near_distance,
                              self.far_distance)

        if len(self.save_dir) > 0:
            save_depth_colormap(
                "%s/%05d_input_depth.png" % (self.save_dir, self.iter),
                depth_image, self.near_distance, self.far_distance)

        # Calculate normals before adding noise
        if self.rgbd_normal_limit > 0.:
            gtNormalImage = np.absolute(
                cv2.Scharr(depth_image, cv2.CV_32F, 1, 0)) + \
                np.absolute(cv2.Scharr(depth_image, cv2.CV_32F, 0, 1))
            _, normalThresh = cv2.threshold(
                gtNormalImage, self.rgbd_normal_limit,
                1., cv2.THRESH_BINARY_INV)

        if self.rgbd_noise > 0.0:
            noiseMat = np.random.randn(h, w)*self.rgbd_noise
            depth_image += noiseMat

        if self.rgbd_projector_baseline > 0.0:

            K = self.camera.depth_camera_info().intrinsic_matrix()
            x_projection = (self.xy1_indices_im[:, :, 0] - K[0, 2]) * \
                depth_image / K[0, 0]

            # For a fixed shift...
            mask = np.ones(depth_image.shape)
            for shift_amt in range(-50, 0, 10):
                imshift_tf_matrix = np.array(
                    [[1., 0., shift_amt], [0., 1., 0.]])
                sh_x_projection = cv2.warpAffine(
                    x_projection, imshift_tf_matrix,
                    (w, h), borderMode=cv2.BORDER_REPLICATE)
                shifted_gt_depth = cv2.warpAffine(
                    depth_image, imshift_tf_matrix, (w, h),
                    borderMode=cv2.BORDER_REPLICATE)

                # (projected test point - projected original point) dot
                # producted with vector perpendicular to sample point
                # and projector origin
                error_im = (sh_x_projection - x_projection)*(-depth_image) + \
                    (shifted_gt_depth - depth_image) * \
                    (x_projection - self.rgbd_projector_baseline)

                # TODO, fix this hard-convert-back-to-32bit-float
                # Threshold any positive error as occluded
                _, error_thresh = cv2.threshold(
                    error_im.astype(np.float32), 0., 1.,
                    cv2.THRESH_BINARY_INV)
                mask *= error_thresh
            depth_image *= mask

            if len(self.save_dir) > 0:
                save_image_colormap(
                    "%s/%05d_mask.png" % (self.save_dir, self.iter), mask)
                save_depth_colormap(
                    "%s/%05d_prerectified_masked_depth.png" % (
                        self.save_dir, self.iter),
                    depth_image, self.near_distance,
                    self.far_distance)

        # Apply normal limiting
        if self.rgbd_normal_limit > 0.:
            depth_image *= normalThresh

        # And finally apply rectification to RGB frame
        if self.rgbd_rectification_baseline != 0.:
            # Convert depth image to point cloud, with +z being
            # camera "forward"
            K = self.camera.depth_camera_info().intrinsic_matrix()
            K_rgb = self.camera.color_camera_info().intrinsic_matrix()
            Kinv = np.linalg.inv(K)
            U, V = np.meshgrid(np.arange(h), np.arange(w))
            points_in_camera_frame = np.vstack([
                U.T.flatten(),
                V.T.flatten(),
                np.ones(w*h)])
            points_in_camera_frame = Kinv.dot(points_in_camera_frame) * \
                depth_image.flatten()
            # Shift them over into the rgb camera frame
            # points_in_camera_frame[0, :] += self.rgbd_rectification_baseline
            # Reproject back into the the image (using the RGB
            # projection matrix. This is wrong (and very bad) if the
            # resolution of the RGB and Depth are different.
            points_in_camera_frame[1, :] += self.rgbd_rectification_baseline
            points_in_camera_frame[0, :] /= points_in_camera_frame[2, :]
            points_in_camera_frame[1, :] /= points_in_camera_frame[2, :]
            points_in_camera_frame[2, :] /= points_in_camera_frame[2, :]
            depth_image_out = np.full(depth_image.shape, np.inf)
            reprojected_uv = K_rgb.dot(points_in_camera_frame)
            for u in range(h):
                for v in range(w):
                    if not np.isfinite(reprojected_uv[2, u*w+v]):
                        continue
                    proj_vu = np.round(reprojected_uv[:, u*w+v]).astype(int)
                    if (proj_vu[0] >= 0 and proj_vu[0] < h and
                            proj_vu[1] >= 0 and proj_vu[1] < w):
                        depth_image_out[proj_vu[0], proj_vu[1]] = (
                            min(depth_image_out[proj_vu[0], proj_vu[1]],
                                depth_image[u, v]))
            # Resaturate infs to 0
            depth_image_out[np.isinf(depth_image_out)] = 0.

        else:
            depth_image_out = depth_image

        if len(self.save_dir) > 0:
            save_depth_colormap(
                "%s/%05d_masked_depth.png" % (
                    self.save_dir, self.iter),
                depth_image_out, self.near_distance,
                self.far_distance)

        # Where it's infinite, set to 0
        depth_image_out = np.where(
            good_mask, depth_image_out,
            np.zeros(depth_image.shape))

        y_data.get_mutable_value().mutable_data[:, :, 0] = \
            depth_image_out[:, :]
        print "Elapsed in render (model): %f seconds" % \
            (time.time() - start_time)
        self.iter += 1
