import advanced_lane_lines.util as util
import numpy as np
import cv2

mask_region = util.get_config_tuples('mask_region')

def mask_image(image):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    height = image.shape[0]
    width = image.shape[1]

    bl = mask_region.get('bottom_left')
    tl = mask_region.get('top_left')
    tr = mask_region.get('top_right')
    br = mask_region.get('bottom_right')

    vertices = np.array([
        [
            (bl[0] * width, bl[1] * image.shape[0]),
            (tl[0] * width, tl[1] * height),
            (tr[0] * width, tr[1] * height),
            (br[0] * width, br[1] * image.shape[0])
        ]
    ], dtype=np.int32)

    #defining a blank mask to start with
    mask = np.zeros_like(image)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image, mask
