import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import matplotlib.pyplot as plt
from collections import deque

class Line():
    """ Class to receive the characteristics of each line detection """

    def __init__(self):

        self.n = 5

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

        ally, allx = image.nonzero()
        try:
            fit = np.polyfit(ally, allx, 2)
            # weight data according to how much of a y spread we have
            # less information affects averages less
            weight = np.max(ally) - np.min(ally)
            self.weights.append(weight)
        except Exception as e:
            fit = self.current_fit
            self.weights.append(self.weights[-1])

        self.recent_fit.append(fit)
        self.current_fit = np.average(self.recent_fit, axis=0, weights=self.weights)

        fx = lambda y, p: p[0] * y ** 2 + p[1] * y + p[2]

        # generate new data from the fit
        ally, allx = [], []
        for yi in np.arange(image.shape[0], 0.0 * image.shape[0], -10):
            xi = fx(yi, self.current_fit)
            ally.append(yi)
            allx.append(xi)

        ally, allx = np.array(ally), np.array(allx)
        fitx = fx(ally, fit)

        self.recent_xfit.append(fitx)
        fitx = np.average(self.recent_xfit, axis=0, weights=self.weights)

        self.ally = ally
        self.allx = allx
        self.current_xfit = fitx

        # decay weights as we get further away from frame
        self.weights = deque(np.multiply(self.weights, 0.3), maxlen=self.n)

        return ally, allx, fitx

    def get_curvature(self):
        # Define y-value where we want radius of curvature
        y_eval = np.max(self.ally)
        fit = self.current_xfit
        curverad = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2*fit[0])
        # Example values: 1163.9    1213.7
        return curverad

    def get_curvature_radians(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        fit_cr = np.polyfit(yvals * ym_per_pix, xvals * xm_per_pix, 2)
        curverad = ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
        # Now our radius of curvature is in meters
        # Example values: 3380.7 m    3189.3 m

    def plot(self, color='red'):
        plt.plot(self.allx, self.ally, 'o', color=color)
        plt.plot(self.current_xfit, self.ally, color='green', linewidth=3)
