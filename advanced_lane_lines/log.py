from configparser import ConfigParser
from os.path import basename, splitext
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

config = ConfigParser()
config.read('config.cfg')
frames = int(config.get('logging', 'frames'))

class Logger():

    logging = False
    mode = None
    frame = 0
    source = None

    # unmodified image is kept for comparison
    unmodified = None

    # the undistorted image is useful in context to perspective transforms
    # where the image being modified is thresholded
    undistorted = None

    @staticmethod
    def reset():
        Logger.frame = 0

    @staticmethod
    def increment():
        Logger.frame += 1

    @staticmethod
    def save(image, name):

        assert Logger.mode is not None, "mode is not set [video, test]"

        # convert binary images to color before saving
        if len(image.shape) == 2:
            image = image.reshape(image.shape + (1,)).astype('uint8') * 255
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # for video mode, save every x frames
        if Logger.mode == 'video' and Logger.frame % frames != 0:
            return

        fname = 'output_images/' + Logger.mode + '/'

        # if image/video source is given use as prefix
        if Logger.source:
            source = splitext(basename(Logger.source))[0]
            fname += source + '-'

        fname += name

        # if video mode, include the frame number
        if Logger.mode == 'video':
            fname += '-' + str(Logger.frame)

        fname += '.jpg'

        log_image = np.concatenate((Logger.unmodified, image), axis=1)
        mpimg.imsave(fname, log_image)
