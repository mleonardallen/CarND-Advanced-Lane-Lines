import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import advanced_lane_lines.util as util

line_angles = util.get_config_tuples('line_angles')

def transform(image, M):
    image_size = (image.shape[1], image.shape[0])
    return cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)

def draw_perspective_lines(image, pts):
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

def get_matrix(src, dest):
    M = cv2.getPerspectiveTransform(
        np.array([src.get('tl'), src.get('tr'), src.get('bl'), src.get('br')], np.float32), 
        np.array([dest.get('tl'), dest.get('tr'), dest.get('bl'), dest.get('br')], np.float32), 
    )
    return M

def line_intersection(line1, line2):

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

def get_transform_points(image):
    # first get the lines
    hough_lines = hough_transform(image)
    left = get_average_line(image.shape, hough_lines, line_angles.get('left'))
    right = get_average_line(image.shape, hough_lines, line_angles.get('right'))

    if not len(left):
        pass
    if not len(right):
        pass

    lslope = get_slope(np.array(left).reshape(4))
    rslope = get_slope(np.array(right).reshape(4))
    intersection = line_intersection(left, right)

    lb = intersection[1] - lslope * intersection[0]
    rb = intersection[1] - rslope * intersection[0]

    backoff = 25
    yi =  intersection[1] + backoff
    xl = int((yi - lb) / lslope)
    xr = int((yi - rb) / rslope)

    src = {
        'tl': (xl, yi),
        'tr': (xr, yi),
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

def hough_transform(image):
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 40    # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 15 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def get_average_line(image_size, hough_lines, limit):

    # top and bottom point same for both lines
    ybottom = image_size[0]
    ytop = np.min(hough_lines.reshape(-1,2)[:,1])

    slopes = np.apply_along_axis(get_slope, 2, hough_lines)

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

def get_slope(line):
    x1,y1,x2,y2 = line
    slope = ((y2-y1)/(x2-x1))
    return slope
