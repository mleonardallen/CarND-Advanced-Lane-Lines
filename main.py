from advanced_lane_lines.pipeline import pipeline
from advanced_lane_lines.log import Logger
from advanced_lane_lines.calibration import calibrate_camera
from moviepy.editor import VideoFileClip
import argparse
import glob
import cv2
from matplotlib.image import imread

def main(mode=None, source=None, out=None, log=False):

    # log properties
    Logger.logging = log
    Logger.mode = mode

    if mode == 'test_images':
        images = glob.glob('test_images/*.jpg')
        for idx, fname in enumerate(images):
            Logger.source = fname
            image = pipeline(imread(fname))
    elif mode == 'video':
        Logger.source = source
        source_video = VideoFileClip(source)
        output_video = source_video.fl_image(pipeline)
        output_video.write_videofile(out, audio=False)
    elif mode == 'calibrate':
        calibrate_camera()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--mode', nargs='?', default='test_images',
        choices=['test_images', 'video', 'calibrate'],
        help='Calibrate camera or run pipeline on test images or video')
    parser.add_argument('--source', nargs='?', default='project_video.mp4', help='Input video')
    parser.add_argument('--out', nargs='?', default='out.mp4', help='Output video')
    parser.add_argument('--log', action='store_true', help='Log output images')
    args = parser.parse_args()

    main(mode=args.mode, source=args.source, out=args.out, log=args.log)