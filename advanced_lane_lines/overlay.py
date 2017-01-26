import numpy as np
import cv2

def draw(image, left_yvals, right_yvals, left_fitx, right_fitx):

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, left_yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the blank image
    image = cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    return image
