import numpy as np
import cv2


class Overlay():


    def __init__(self):
        self.white = (255, 255, 255)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_position = (20, 40)

    def draw(self, image, left, right):

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left.current_xfit, left.ally]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right.current_xfit, right.ally])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the blank image
        image = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        return image

    def stats(self, image, left, right):
        self.text_position = (20, 40)

        right_curvature = right.get_curvature()
        left_curvature = left.get_curvature()
        average_curvature = np.mean([left_curvature, right_curvature])
        threshold = 1000


        if -threshold <= left_curvature <= threshold:
            self.putText(image, "Left Curvature: {:.2f} m".format(left_curvature))
        else:
            self.putText(image, "Left Curvature: Straight")

        if -threshold <= right_curvature <= threshold:
            self.putText(image, "Right Curvature: {:.2f} m".format(right_curvature))
        else:
            self.putText(image, "Right Curvature: Straight")

        # left distance will be negative
        left_distance = left.get_distance_from_center()
        right_distance = right.get_distance_from_center()
        distance = left_distance + right_distance

        if distance > 0:
            self.putText(image, "Distance from Center: {:.2f} m to the right".format(distance))
        elif distance < 0:
            self.putText(image, "Distance from Center: {:.2f} m to the left".format(np.abs(distance)))
        else:
            self.putText(image, "Distance from Center: 0 m")

        return image

    def putText(self, image, text):
        cv2.putText(image, text, self.text_position, self.font, 1, self.white, 2, cv2.LINE_AA)
        self.text_position = (self.text_position[0], self.text_position[1] + 35)
