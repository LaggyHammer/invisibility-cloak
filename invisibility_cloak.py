# coding: utf-8
# =====================================================================
#  Filename:    image_orientation.py
#
#  py Ver:      python 3.6 or later
#
#  Description: Using opencv masks, the module renders objects of a given color invisible to the camera.
#  Run the invisibility_cloak script and let it see the background for a few seconds.
#  Come in with the object/cloak into the frame to render it invisible.
#
#  Usage: You need to specify a filter or the upper & lower bounds for a filter in the HSV colorspace (comma separated)
#  python invisibility_cloak.py --filter red
#  or
#  python invisibility_cloak.py --filter custom --lower 10,150,0 --upper 140,255,255
#
#  Author: Ankit Saxena (ankch24@gmail.com)
# =====================================================================


import cv2
import numpy as np
import time
import argparse


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filter', required=True, default='red',
                    help='color of filter in HSV colorspace')
    ap.add_argument('-l', '--lower', required=False,
                    help='lower range of filter in HSV colorspace')
    ap.add_argument('-u', '--upper', required=False,
                    help='upper range of filter in HSV colorspace')
    arguments = vars(ap.parse_args())

    return arguments


def color_mask(hsv, color, upper, lower):
    """
    Given a color OR an upper/lower range in the HSV color space, generate a mask for the same.
    :param hsv: image in HSV color space
    :param color: color for mask
    :param upper: upper bound for HSV color mask range
    :param lower: lower bound for HSV color mask range
    :return: mask of desired color/range
    """
    if color is not None:

        if color.lower() == 'blue':
            lower = np.array([10, 150, 0])
            upper = np.array([140, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            return mask

        if color.lower() == 'red':
            lower = np.array([0, 125, 50])
            upper = np.array([10, 255, 255])
            mask_1 = cv2.inRange(hsv, lower, upper)

            lower = np.array([170, 120, 70])
            upper = np.array([180, 255, 255])
            mask_2 = cv2.inRange(hsv, lower, upper)

            return mask_1 + mask_2

    elif upper is not None and lower is not None:
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)

        return mask


def save_background(video_feed):
    """
    Waits for 3 seconds then returns the 60th web-cam feed frame as the background frame
    :param video_feed: video capture feed to be used
    :return: frame to be used as background
    """
    time.sleep(3)

    background = 0

    for i in range(60):
        _, background = video_feed.read()
    background = np.flip(background, axis=1)

    return background


def create_videowriter(file_name, fps, size):
    """
    Creates a video writer object to save the video feed module results
    :param file_name: file to save the video in
    :param fps: frames per second for the video
    :param size: size/resolution of the video feed
    :return: video writer object to write to
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file_name, fourcc, fps, size)

    return out


def main(file_name='output.avi', fps=20.0, size=(640, 480), color=None, upper=None, lower=None):

    # Define video writer
    out = create_videowriter(file_name, fps, size)

    # Define video feed
    cap = cv2.VideoCapture(0)

    # Saving background
    background = save_background(video_feed=cap)

    count = 0

    # Main loop
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        count += 1

        # flip image to show mirrored image
        img = np.flip(img, axis=1)

        # change to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # get color mask
        mask1 = color_mask(hsv, color, upper, lower)

        # open & dilate the mask to make mask clearer
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # background mask
        mask2 = cv2.bitwise_not(mask1)

        # invisibility cloak
        res1 = cv2.bitwise_and(img, img, mask=mask2)

        # background
        res2 = cv2.bitwise_and(background, background, mask=mask1)

        # adding both results & displaying
        final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
        out.write(final_output)
        cv2.imshow("Invisibility Cloak", final_output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    # releasing the video capture & closing windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_arguments()
    range_filter = args['filter'].lower()
    if range_filter == 'custom':

        main(upper=list(map(int, args['upper'].split(','))),
             lower=list(map(int, args['lower'].split(','))))

    else:
        main(color=range_filter)



