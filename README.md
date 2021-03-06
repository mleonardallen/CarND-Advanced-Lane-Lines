## Advanced Lane Finding Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

```
python main.py

--mode [{test_images,video,calibrate}]
                      Calibrate camera or run pipeline on test images or video
--source [SOURCE]     Input video
--out [OUT]           Output video
--log                 Log output images
```

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrate` method located in `advanced_lane_lines/calibration.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.  Because not all inner corners are visible in each image, I produce mutiple object points.  With each image I try to detect the most corners and then fall back until as many corners as possible are found.

```
sizes = [(9,6), (8,6), (9,5), (7,6)]
```

If `cv2.findChessboardCorners()` detects the given size, the `objp` coordinates for the given size is appended to `objpoints`.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  The resulting matrix and distortion coefficients are then stored in a pickle file at `camera_cal/calibration.p`

I applied this distortion correction to the camera calibration image using the `cv2.undistort()` function and obtained this result: 

##### Distorted
![Distorted](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/camera_cal/calibration1.jpg)

##### Undistorted
![Undistorted](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/calibrate/calibration1-16-undistored.jpg)

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![Original](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/test_images/solidWhiteRight-01-original.jpg)

The first step in the pipeline (method `process` in `advanced_lane_lines/pipeline.py`) is to undistort original image.  To do this we leverage the saved calibration data and `cv2.undistort()` (method `undistort` in `advanced_lane_lines/calibration.py`).

```
image = calibration.undistort(image)
```

![Undistorted](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/test_images/solidWhiteRight-02-undistorted.jpg)

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (method `combined_thresh` in `advanced_lane_lines/threshold.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

##### Sobel X Binary
First, I take a sobel threshold in the x direction.  This limits vertical gradients which would not contain lane lines in our test image or project video.  I decided to go with this instead of a more general magnitude and direction combination because I would essentially be trying to reproduce sobel x.

![Sobel X Binary](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/video/project_video-600-03-sobelx-binary.jpg)

##### Saturation Binary

Sobel thresholds have trouble in some scenarios, such as a yellow line on a brown road.  Saturation is better at picking out yellow lines in these scenarios.

![Saturation Binary](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/video/project_video-600-04-saturation-binary.jpg)

##### Lightness Binary

A lightness binary is used to remove extrenous pixels contained in the saturation and sobel binaries.  The idea here is that the dark parts picked up by saturation or sobel are either shadow or other marks in the road unrelated to lane lines.

![Lightness Binary](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/video/project_video-600-05-lightness-binary.jpg)

##### Combined Binary

This binary is a combination of `sobel x` or `saturation`, and `lightness` thresholded binaries.

![Combined Binary](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/video/project_video-600-08-combined-binary.jpg)

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Because each test image and video has different resolution and camera mount setup, I chose to not hardcode the source and destination points.  Instead, to determine these points, I repurposed lane line detection code from my [CarND-LaneLines-P1 Project](https://github.com/mleonardallen/CarND-LaneLines-P1).  Although we will draw these lane lines on our final output display, we can use them as a good approximation for where to do the perspective transoform using these lines.

The code for my perspective transform includes a function called `get_transform_points()`, which appears in the file `advanced_lane_lines/perspective.py`.  The `get_transform_points()` function takes as inputs an image (`img`), and returns source (`src`) and destination (`dest`) points.

Example resulting source and destination points (changes for each image):

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 1116, 720      | 1116, 720        | 
| 276, 720      | 276, 720      |
| 619, 469     | 276, 0      |
| 722, 469      | 1116, 0        |

##### Perspective Transform Step #1: Mask Image
Mask image to focus on area of the image that contains lane lines (method `mask_image` in `advanced_lane_lines/mask.py`)

![Masked](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/video/project_video-600-09-masked-image.jpg)

##### Perspective Transform Step #2: Hough Lines and Average
Using hough transform (method `hough_transform` in `perspective.py`), I detect lines within the masked binary image.  Hough lines are then sorted into left and right lanes and averaged (method `get_average_line` in `perspective.py`).

![Hough Lines](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/video/project_video-600-10-hough-lines.jpg)

##### Perspective Transform Step #3: Vanishing Point

Now that we have average left and right lines, we can calculate the source points.  The bottom points will be the origin of the lines at the bottom of the image.  To determine the top points we will leverage the [Vanishing Point](https://en.wikipedia.org/wiki/Vanishing_point).  Parallel lines converge at a vanishing point, and since lane lanes are parellel we assume that they will converge at a vanishing point.  Another assumption we make is that we are dealing with relatively flat roads.

Using a technique described in the [Udacity Forums](https://carnd-forums.udacity.com/cq/viewquestion.action?id=29494501&answerId=34575350), I determine the vanishing point (method `line_intersection` in `perspective.py`), and then back off a little for the top points.  In experimenting, I found that getting too close to the vanishing point gave an increasingly blurry transformation.

##### Perspective Transform Step #4: Debug Output

_Note: In the actual source image at this step is a binary thresholded image.  I am using the undistorted image here in this example instead of the binary thresholded image because I found it more useful for visualizing the transformation._

Destination points are relatively simple compared to the source points.  Basically the destination points are at the top of the image have same x value as the points at the bottom.  The bottom points are the same as in the source points.

![Source Points](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/video/project_video-600-11-perspective-transform-src.jpg)

![Destination Points](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/video/project_video-600-12-perspective-transform-dest.jpg)

##### Perspective Transform Step #5: Transform to Birds-Eye View

I verified that my perspective transform was working as expected by drawing the `src` and `dest` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warped Image](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/video/project_video-600-13-perspective-transform-binary.jpg)

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

##### Lane Line Pixels Step #1: Histogram
The first step in detecting lane line pixels is to take a histogram of the birds-eye view thresholded image (method `get_histogram` in `advanced_lane_lines/lane_finder.py`)

```
histogram = np.sum(image[image.shape[0]/2:,:], axis=0)
```

The resulting peaks in the histogram are great indicators for where to start looking for lane line pixels.

![Histogram](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/video/project_video-600-14-histogram.jpg)

##### Lane Line Pixels Step #2: Window Seek

Using the starting position from the histogram view the image through a small window.  See method `get_lane_pixels` in `advanced_lane_lines/lane_finder.py`.  Taking an average of the x values within the window will tell us which direction to move the window in for our next slice as we go up the image looking for lane line pixels.

![Window Seek](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/video/project_video-600-15-lane-seek.jpg)

##### Lane Line Pixels Step #3: Fit Polynomial

In order to draw the lane overlay, we first need to fit a polynomial using the left and right lane pixels (method `fit` in `advanced_lane_lines/line.py`).  This function takes lane pixels and fits a 2nd order polynomial.

In order to account for negative scenarios, if we did not detect lane pixels, a previous fit is leveraged.

Also, in order to have smoother transitions between frames, an average over the previous `10` frames is leveraged.  The weight of each frame in the average is determined by the spread of y values within the detected pixels as well as a decay as the frame gets further away in time.

_Note: Once we have our polynomial coefficients, I generate new x and y values for later drawing the lane overlay.  This is because we may not have enough detected lane pixels to cover a large area within the drawn overlay.  The generated data more consistantly covers a larger area._

![2nd Order Polynomial](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/video/project_video-600-16-curvature.jpg)

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

For curvature, see `get_curvature` in my code in `advanced_lane_lines/line.py`.  Pixel curvature is converted into meters using conversion values provided by Udacity.  For my purposes, a curvature radious over 1000 m is considered straight.

```
xm_per_pix = 3.7/700 # meteres per pixel in x dimension
ym_per_pix = 30/720 # meters per pixel in y dimension
```

Similarly, distance from center first calculates the distance of each lane from the center of the image, and then converts to meters (method `get_distance_from_center` in `advanced_lane_lines/line.py`).  Since the left distance is negative and the right distance is positive, to get the total distance away from center, we add the left and right distances together.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `advanced_lane_lines/overlay.py` in the function `draw()`, which takes the birds-eye view image and fills in a polygon defined by the left and right line fitted points.

Once the overlay is drawn we then transform back to the original perspective (method `transform` in `advanced_lane_lines/perspective.py`).  This time however, our transform takes the inverse matrix of our original transformation.

Curvature and distance from center text is handled in `stats()`.  Here is an example of my result on a test image:

![Final Result](https://github.com/mleonardallen/CarND-Advanced-Lane-Lines/blob/master/output_images/video/project_video-600-17-final.jpg)

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/TgBc9m3ZesA)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

#### Thresholded Binary Image
This section contained a lot of tweaking of parameters.  I started by extracting values into a config file so that I could adjust values all in one spot.  This helped me when I needed to go back and change parameters.  However, in the end, not all parameters made it into the config file.  Considering the vast number of possible parameters, it is not feasible to a human to search the entire space getting the best possible combinations.  In order improve this, I would want to set up some sort of grid search that includes possible parameter combinations as well binary thresholded combinations.  This introduces the issue of what to use as error metric when determining the best combinations, but perhaps a human could label the correct pixels in each frame, and the loss function could compare the thresholded pixels vs the human labeled pixels.

#### Perspective Transform
Here, information from the previous frames can be used to provide a starting search location.  Instead of using a window seek method, I could use the current fitted polynomial could be leveraged to find the lane pixels in the next frame.  This would be more stable for frames with noisy lane pixels.

This step in the pipeline includes a masking step, however a more sophisticated mask might be leveraged.  Using the previous fitted curve, we could the and area around a single lane line, exluding more extraneous pixels and getting a better estimate for the lane lines during the perspective transform.

After testing my model on the challenge video, I noted that the binary thresholded image did not exclude many crack lines that ran parallel to the lane.  Since these lines fell into my range for an appropriate angle for a lane line, they were included into to sorted left and right arrays for lane pixels.  Better thresholding or masking could exclude these lines.

Another method for a more stable transform step could be to leverage the intrinsic (focal length and optical center) and extrinsic (pitch angle, yaw angle, and height) to perform the bird-eye view transformation.  Since these properties do not change from frame to frame, this would result in a transformation that is stable through trouble frames where the lane lines are not easy to detect.  Method detailed in this [Paper](http://www.vision.caltech.edu/malaa/publications/aly08realtime.pdf)

Another issue that could pose a problem when doing the perspective transform as well as other steps is obstacles such as cars or debris in the road.  For example if a car is crossing a lane, this could make detecting lane lines very difficult.  One possible solution is that we can use sensor fusion to determine which pixels are likely a car or other object.

#### Line Curvature
I chose to take a weighted average to determine line curvature, which results in a more stable curve, but also creates a less accurate fit since the previous frames are causing the current fit to be pulled in the direction of previous frames.  One possible improvement would be to introduce a low-pass filter that is resistant to large changes, but allows small changes to go though.
