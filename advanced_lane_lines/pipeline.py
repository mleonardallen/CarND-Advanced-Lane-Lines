import advanced_lane_lines.calibration as calibration
import advanced_lane_lines.threshold as threshold
import advanced_lane_lines.mask as mask

import advanced_lane_lines.lane_finder as lane_finder
from advanced_lane_lines.log import Logger
from advanced_lane_lines.line import Line
from advanced_lane_lines.perspective import Perspective
from advanced_lane_lines.overlay import Overlay
import matplotlib.pyplot as plt
import copy
import cv2

class Pipeline():

    def __init__(self):
        self.reset()

    def reset(self):
        self.left = Line()
        self.right = Line()
        self.perspective = Perspective()
        self.overlay = Overlay()

    def process(self, image):

        Logger.save(image, 'original')

        # Apply the distortion correction to the raw image.
        image = calibration.undistort(image)
        undistorted_image = copy.copy(image)
        Logger.save(undistorted_image, 'undistorted')

        # Use color transforms, gradients, etc., to create a thresholded binary image.
        thresholded_image = threshold.combined_thresh(image)
        masked_image = mask.mask_image(thresholded_image)

        # Apply a perspective transform to rectify binary image ("birds-eye view").
        src, dest = self.perspective.get_transform_points(masked_image)
        M, Minv = self.perspective.get_matrix(src, dest), self.perspective.get_matrix(dest, src)
        image_src = self.perspective.draw_perspective_lines(undistorted_image, src)
        Logger.save(image_src, 'perspective-transform-src')
        image_dest = self.perspective.transform(undistorted_image, M)
        image_dest = self.perspective.draw_perspective_lines(image_dest, dest)
        Logger.save(image_dest, 'perspective-transform-dest')
        transformed_image = self.perspective.transform(thresholded_image, M)
        Logger.save(transformed_image, 'perspective-transform-binary')

        # Detect lane pixels and fit to find lane boundary.
        left_pixels, right_pixels = lane_finder.get_lane_pixels(transformed_image)

        self.left.fit(left_pixels)
        self.right.fit(right_pixels)

        lane_finder.plot(self.left, self.right, transformed_image.shape)
        lane_boundaries = self.overlay.draw(transformed_image, self.left, self.right)

        # Warp the detected lane boundaries back onto the original image.
        # Combine the result with the original image
        lane_boundaries = self.perspective.transform(lane_boundaries, Minv)
        image = cv2.addWeighted(undistorted_image, 1, lane_boundaries, 0.3, 0)

        # Determine curvature of the lane and vehicle position with respect to center.
        # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
        image = self.overlay.stats(image, self.left, self.right)
        Logger.save(image, 'final')
        Logger.increment()
        return image
