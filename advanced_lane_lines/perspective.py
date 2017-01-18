from advanced_lane_lines.log import Logger
from configparser import ConfigParser
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt


config = ConfigParser()
config.read('config.cfg')


def transform(image):

    src, dest = get_transform_points(image)
    M = cv2.getPerspectiveTransform(
        np.array([src.get('tl'), src.get('tr'), src.get('bl'), src.get('br')], np.float32), 
        np.array([dest.get('tl'), dest.get('tr'), dest.get('bl'), dest.get('br')], np.float32), 
    )
    image_size = (image.shape[1], image.shape[0])
    image = cv2.warpPerspective(image, M, image_size)

    # note: undistored image used for debugging output, not used in final transform
    image_src = copy.copy(Logger.undistort)
    image_src = cv2.line(image_src, src.get('tl'), src.get('bl'), [255, 0, 0], 1)
    image_src = cv2.line(image_src, src.get('tr'), src.get('br'), [255, 0, 0], 1)
    image_src = cv2.line(image_src, src.get('tl'), src.get('tr'), [255, 0, 0], 1)
    # note: undistored image used for debugging output, not used in final transform
    image_dest = copy.copy(Logger.undistort)
    image_dest = cv2.warpPerspective(image_dest, M, image_size, flags=cv2.INTER_LINEAR)
    image_dest = cv2.line(image_dest, dest.get('tl'), dest.get('bl'), [255, 0, 0], 1)
    image_dest = cv2.line(image_dest, dest.get('tr'), dest.get('br'), [255, 0, 0], 1)
    image_dest = cv2.line(image_dest, dest.get('tl'), dest.get('tr'), [255, 0, 0], 1)

    Logger.save(image_src, 'perspective-transform-src')
    Logger.save(image_dest, 'perspective-transform-dest')
    Logger.save(image, 'perspective-transform-binary')

    return image

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def get_average_lines(image, lines):

    # slopes are used to separate lines into right and left side
    slopes = []
    top = 0
    bottom = image.shape[0]

    slopes = np.apply_along_axis(get_slope, 2, lines)
    # top and bottom point same for both lines
    top = np.min(lines.reshape(-1,2)[:,1])

    # sort into sides and average
    avgs = []

    for name, lower, upper in [('left', -1.0, -0.5), ('right', 0.5, 1.0)]:
        # Separate points into left or right side

        side_lines = lines[np.logical_and(slopes >= lower, slopes < upper)]
        if len(side_lines) == 0:
            return []

        # reshape lines into x, y series for polyfit
        side_lines = side_lines.reshape(-1,2)
        x = side_lines[:,0]
        y = side_lines[:,1]
        slope, intercept = np.polyfit(x, y, 1)

        # Averaged line points
        avg = np.array([
            (bottom - intercept) / slope, bottom,
            (top - intercept) / slope, top
        ]).astype(int)

        avgs.append([
            (avg[0], avg[1]),
            (avg[2], avg[3])
        ])

    return avgs

def get_slope(line):
    x1,y1,x2,y2 = line
    slope = ((y2-y1)/(x2-x1))
    return slope

def get_lines(image, average=False):

    # crop image to section containing lines
    height = image.shape[0]
    width = image.shape[1]
    vertices = np.array([
        [
            (0.05 * width, image.shape[0]),
            (width * 0.43, height * 0.60),
            (width * 0.58, height * 0.60),
            (0.95 * width, image.shape[0])
        ]
    ], dtype=np.int32)
    image = region_of_interest(image, vertices)
    Logger.save(image, 'perspective-region-mask')

    # hough transform
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 40    # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 15 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    lines = get_average_lines(image, lines)
    return lines

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
    left, right = get_lines(image)
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
