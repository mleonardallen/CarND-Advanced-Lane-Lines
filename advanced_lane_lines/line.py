import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import matplotlib.pyplot as plt
from collections import deque

class Line():
    """ Class to receive the characteristics of each line detection """

    def __init__(self):

        self.n = 10

        # was the line detected in the last iteration?
        self.detected = False

        self.weights = deque(maxlen = self.n)

        # x values of the last n fits of the line
        self.recent_xfit = deque(maxlen = self.n)

        self.current_xfit = None

        #polynomial coefficients of the last n iterations
        self.recent_fit = deque(maxlen = self.n)

        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        #radius of curvature of the line in some units
        self.radius_of_curvature = None

        #distance in meters of vehicle center from the line
        self.line_base_pos = None

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        #x values for detected line pixels
        self.allx = None

        #y values for detected line pixels
        self.ally = None

    def fit(self, image):

        originaly, originalx = image.nonzero()
        try:
            fit = np.polyfit(originaly, originalx, 2)
            # weight data according to how much of a y spread we have
            # less information affects averages less
            weight = np.max(originaly) - np.min(originaly)
            self.weights.append(weight)
        except Exception as e:
            fit = self.current_fit
            self.weights.append(self.weights[-1])

        self.recent_fit.append(fit)
        self.current_fit = np.average(self.recent_fit, axis=0, weights=self.weights)

        fx = lambda y, p: p[0] * y ** 2 + p[1] * y + p[2]

        # generate new data from the fit so we can visually extend the overlay beyond the data
        ally, allx = [], []
        for yi in np.arange(image.shape[0], 0, -10):
            xi = fx(yi, self.current_fit)
            ally.append(yi)
            allx.append(xi)

        ally, allx = np.array(ally), np.array(allx)
        fitx = fx(ally, fit)

        # starting point of the line
        self.origin = allx[0]
        self.height = image.shape[0]
        self.width = image.shape[1]

        self.recent_xfit.append(fitx)
        fitx = np.average(self.recent_xfit, axis=0, weights=self.weights)

        # decay weights as we get further away from frame
        self.weights = deque(np.multiply(self.weights, 0.4), maxlen=self.n)

        # store both generated and original x/y values
        # both are needed for plotting and or overlays
        self.originaly = originaly
        self.originalx = originalx

        self.ally = ally
        self.allx = allx

        self.current_xfit = fitx

    def get_curvature(self, meters = True):
        """ get line curvature """

        y = self.ally.astype('float64')
        x = self.allx.astype('float64')

        # final curvature is in meters
        if meters:
            x = self.pixels_to_meters(x, dimension='x')
            y = self.pixels_to_meters(y, dimension='y')

        y_eval = np.max(y)
        fit = np.polyfit(y, x, 2)

        curvature = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2*fit[0])
        return curvature

    def get_distance_from_center(self):
        distance = int(self.origin - self.width / 2)
        return self.pixels_to_meters(distance, dimension='x')

    def pixels_to_meters(self, vals, dimension='x'):
        """ If radians, define conversions in x and y from pixels space to meters """
        if dimension == 'x':
            xm_per_pix = 3.7/700 # meteres per pixel in x dimension
            vals *= xm_per_pix
        if dimension == 'y':
            ym_per_pix = 30/720 # meters per pixel in y dimension
            vals *= ym_per_pix

        return vals

    def plot(self, color='red'):
        plt.plot(self.originalx, self.originaly, 'o', color=color)
        plt.plot(self.current_xfit, self.ally, color='green', linewidth=3)
