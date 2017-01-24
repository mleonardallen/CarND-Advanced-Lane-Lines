from advanced_lane_lines.calibration import undistort
from advanced_lane_lines.threshold import combined_thresh
from advanced_lane_lines.perspective import transform, untransform
from advanced_lane_lines.curvature import get_curvature_values, draw
from advanced_lane_lines.log import Logger

def pipeline(image):
    Logger.unmodified = image

    # Apply the distortion correction to the raw image.
    image = undistort(image)
    # Use color transforms, gradients, etc., to create a thresholded binary image.
    image = combined_thresh(image)
    # Apply a perspective transform to rectify binary image ("birds-eye view").
    image, M, Minv = transform(image)
    # Detect lane pixels and fit to find lane boundary.
    # scatterplot()
    left_yvals, right_yvals, left_fitx, right_fitx = get_curvature_values(image)
    image = draw(image, left_yvals, right_yvals, left_fitx, right_fitx)
    # Warp the detected lane boundaries back onto the original image.
    image = untransform(image, Minv)
    
    # Determine curvature of the lane and vehicle position with respect to center.
    
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    Logger.increment()
    return image
