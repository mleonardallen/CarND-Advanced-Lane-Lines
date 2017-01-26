import advanced_lane_lines.calibration as calibration
import advanced_lane_lines.threshold as threshold
import advanced_lane_lines.perspective as perspective
import advanced_lane_lines.mask as mask
import advanced_lane_lines.overlay as overlay
import advanced_lane_lines.lane_finder as lane_finder
from advanced_lane_lines.log import Logger
from advanced_lane_lines.line import Line
import copy
import cv2

left, right = Line(), Line()

def reset():
    left, right = Line(), Line()

def process(image):

    # Apply the distortion correction to the raw image.
    image = calibration.undistort(image)
    undistorted_image = copy.copy(image)
    Logger.save(undistorted_image, 'undistorted')

    # Use color transforms, gradients, etc., to create a thresholded binary image.
    thresholded_image = threshold.combined_thresh(image)
    Logger.save(thresholded_image, 'combined-binary')
    masked_image, image_mask = mask.mask_image(thresholded_image)
    Logger.save(masked_image, 'masked-image')

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    src, dest = perspective.get_transform_points(masked_image)
    M, Minv = perspective.get_matrix(src, dest), perspective.get_matrix(dest, src)
    image_src = perspective.draw_perspective_lines(undistorted_image, src)
    Logger.save(image_src, 'perspective-transform-src')
    image_dest = perspective.transform(undistorted_image, M)
    image_dest = perspective.draw_perspective_lines(image_dest, dest)
    Logger.save(image_dest, 'perspective-transform-dest')
    transformed_image = perspective.transform(thresholded_image, M)
    Logger.save(transformed_image, 'perspective-transform-binary')

    # Detect lane pixels and fit to find lane boundary.
    left_pixels, right_pixels = lane_finder.get_lane_pixels(transformed_image)
    left_yvals, left_xvals, left_fitx = left.fit_line(left_pixels)
    right_yvals, right_xvals, right_fitx = right.fit_line(right_pixels)
    lane_finder.plot(left, right, transformed_image.shape)
    lane_boundaries = overlay.draw(transformed_image, left_yvals, right_yvals, left_fitx, right_fitx)

    # Warp the detected lane boundaries back onto the original image.
    # Combine the result with the original image
    lane_boundaries = perspective.transform(lane_boundaries, Minv)
    image = cv2.addWeighted(undistorted_image, 1, lane_boundaries, 0.3, 0)
    Logger.save(image, 'lane-boundaries')

    # Determine curvature of the lane and vehicle position with respect to center.


    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    Logger.increment()
    return image
