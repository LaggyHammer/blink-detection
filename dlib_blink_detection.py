import argparse
import cv2
import dlib
import imutils
import numpy as np
import time
from imutils.video import FPS
from imutils import face_utils
from scipy.spatial import distance


def eye_aspect_ratio(eye):

    vertical_1 = distance.euclidean(eye[1], eye[5])
    vertical_2 = distance.euclidean(eye[2], eye[4])
    horizontal = distance.euclidean(eye[0], eye[3])

    ratio = (vertical_1 + vertical_2) / (2 * horizontal)

    return ratio


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', type=str,
                    help='video stream to detect blinks in')
    ap.add_argument('-p', '--shape_predictor', type=int, default=20,
                    help='path to facial landmark predictor')
    arguments = vars(ap.parse_args())

    return arguments


if __name__ == '__main__':

    args = get_arguments()
