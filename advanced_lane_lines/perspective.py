from advanced_lane_lines.log import Logger
from configparser import ConfigParser
import cv2
import numpy as np
import copy

config = ConfigParser()
config.read('config.cfg')
perspective = config.items('perspective')
perspective = {item[0]: tuple(map(float, item[1].split())) for item in perspective}

src_percent = np.float32([
    perspective.get('src_top_right'),
    perspective.get('src_bottom_right'),
    perspective.get('src_bottom_left'),
    perspective.get('src_top_left'),
])
dest_percent = np.float32([
    perspective.get('dest_top_right'),
    perspective.get('dest_bottom_right'),
    perspective.get('dest_bottom_left'),
    perspective.get('dest_top_left')
])

def transform(image):
    image_size = (image.shape[1], image.shape[0])
    src, dest = get_coords(image_size)

    image_src = copy.copy(Logger.undistort)
    for pt in src:
        image_src = cv2.circle(image_src, tuple(pt), 5, (0, 0, 255), -1)
    Logger.save(image_src, 'perspective-src-points')

    image_dest = copy.copy(Logger.undistort)
    for pt in dest:
        image_src = cv2.circle(image_dest, tuple(pt), 5, (0, 0, 255), -1)
    Logger.save(image_dest, 'perspective-dest-points')

    M = cv2.getPerspectiveTransform(src, dest)
    image = cv2.warpPerspective(image, M, image_size)
    Logger.save(image, 'perspective-transform')

    return image

def get_coords(image_size):
    src = np.array([np.multiply(item, image_size) for item in src_percent], np.float32)
    dest = np.array([np.multiply(item, image_size) for item in dest_percent], np.float32)
    return src, dest