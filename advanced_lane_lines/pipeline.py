from advanced_lane_lines.calibration import undistort
from advanced_lane_lines.threshold import combined_thresh
from advanced_lane_lines.perspective import transform
from advanced_lane_lines.log import Logger

def pipeline(image):
    Logger.unmodified = image

    # Apply the distortion correction to the raw image.
    image = undistort(image)
    # Use color transforms, gradients, etc., to create a thresholded binary image.
    image = combined_thresh(image)
    # Apply a perspective transform to rectify binary image ("birds-eye view").
    image = transform(image)

    # Detect lane pixels and fit to find lane boundary.
    # Determine curvature of the lane and vehicle position with respect to center.
    # Warp the detected lane boundaries back onto the original image.
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

    Logger.increment()
    return image
