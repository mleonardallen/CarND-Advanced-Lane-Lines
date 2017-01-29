import cv2
import numpy as np
import copy
import advanced_lane_lines.util as util
from advanced_lane_lines.log import Logger

class Perspective():
    """ Class to manage perspective transform """

    def __init__(self):
        self.line_angles = util.get_config_tuples('line_angles')

        # line approximations for perspective transform
        self.left = None
        self.right = None

    def transform(self, image, M):
        image_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)

    def draw_perspective_lines(self, image, pts):
        image = copy.copy(image)
        tl, tr, bl, br = [
            tuple(map(int, pts.get('tl'))),
            tuple(map(int, pts.get('tr'))),
            tuple(map(int, pts.get('bl'))),
            tuple(map(int, pts.get('br')))
        ]

        image = cv2.line(image, tl, bl, [255, 0, 0], 1)
        image = cv2.line(image, tr, br, [255, 0, 0], 1)
        image = cv2.line(image, tl, tr, [255, 0, 0], 1)
        return image

    def get_matrix(self, src, dest):
        M = cv2.getPerspectiveTransform(
            np.array([src.get('tl'), src.get('tr'), src.get('bl'), src.get('br')], np.float32), 
            np.array([dest.get('tl'), dest.get('tr'), dest.get('bl'), dest.get('br')], np.float32), 
        )
        return M

    def line_intersection(self, line1, line2):

        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
           raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return int(x), int(y)

    def get_transform_points(self, image):
        # first get the lines
        hough_lines = self.hough_transform(image)

        left = self.get_average_line(image.shape, hough_lines, self.line_angles.get('left'))
        right = self.get_average_line(image.shape, hough_lines, self.line_angles.get('right'))

        if not len(left):
            left = self.left
        if not len(right):
            right = self.right

        self.left = left
        self.right = right

        lslope = self.get_slope(np.array(left).reshape(4))
        rslope = self.get_slope(np.array(right).reshape(4))

        intersection = self.line_intersection(left, right)

        left_b = intersection[1] - lslope * intersection[0]
        right_b = intersection[1] - rslope * intersection[0]

        backoff = 35
        top_y =  intersection[1] + backoff

        left_top_x = int((top_y - left_b) / lslope)
        right_top_x = int((top_y - right_b) / rslope)

        src = {
            'tl': (left_top_x, top_y),
            'tr': (right_top_x, top_y),
            'bl': (left[0][0], left[0][1]),
            'br': (right[0][0], right[0][1]),
        }

        dest = {
            'tl': (left[0][0], 0),
            'tr': (right[0][0], 0),
            'bl': (left[0][0], left[0][1]),
            'br': (right[0][0], right[0][1]),
        }

        return src, dest

    def hough_transform(self, image):
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 50    # minimum number of votes (intersections in Hough grid cell)
        min_line_len = 15 #minimum number of pixels making up a line
        max_line_gap = 30    # maximum gap in pixels between connectable line segments
        hough_lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

        # save debug image
        color_image = np.dstack((image, image, image)) * 255
        line_image = np.zeros_like(color_image)

        for line in hough_lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)

        line_image = cv2.addWeighted(color_image, 0.7, line_image, 0.7, 0)
        Logger.save(line_image, 'hough-lines')

        return hough_lines

    def get_average_line(self, image_size, hough_lines, limit):

        # top and bottom point same for both lines
        ybottom = image_size[0]
        ytop = np.min(hough_lines.reshape(-1,2)[:,1])

        slopes = np.apply_along_axis(self.get_slope, 2, hough_lines)

        # Separate points into left or right side
        side_lines = hough_lines[np.logical_and(slopes >= limit[0], slopes < limit[1])]
        if len(side_lines) == 0:
            return []

        # reshape lines into x, y series for polyfit
        side_lines = side_lines.reshape(-1,2)
        x = side_lines[:,0]
        y = side_lines[:,1]
        slope, intercept = np.polyfit(x, y, 1)

        # Averaged line points
        xbottom = (ybottom - intercept) / slope
        xtop = (ytop - intercept) / slope
        return [(xbottom, ybottom), (xtop, ytop)]

    def get_slope(self, line):
        x1,y1,x2,y2 = line
        slope = ((y2-y1)/(x2-x1))
        return slope
