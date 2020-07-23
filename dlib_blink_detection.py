# coding: utf-8
# =====================================================================
#  Filename:    dlib_blink_detection.py
#
#  py Ver:      python 3.6 or later
#
#  Description: Detects how many times a person blinks on a video-stream.
#
#  Usage: python dlib_blink_detection.py --shape_predictor shape_predictor_68_face_landmarks.dat
#         or
#         python dlib_blink_detection.py --video test.mp4 --shape_predictor shape_predictor_68_face_landmarks.dat
#
#  Note: Requires opencv 3.4.2 or later
#
#  Author: Ankit Saxena (ankch24@gmail.com)
# =====================================================================

import argparse
import cv2
import dlib
import imutils
import time
from imutils.video import FPS
from imutils import face_utils
from scipy.spatial import distance


def eye_aspect_ratio(eye):
    """
    Calculates the ratio of the vertical to the horizontal width of the eye detected. A dip in the ratio corresponds
    to a blink
    :param eye: numpy array with eye landmark co-ordinates
    :return: eye aspect ratio
    """
    vertical_1 = distance.euclidean(eye[1], eye[5])
    vertical_2 = distance.euclidean(eye[2], eye[4])
    horizontal = distance.euclidean(eye[0], eye[3])

    ratio = (vertical_1 + vertical_2) / (2 * horizontal)

    return ratio


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', type=str,
                    help='video stream to detect blinks in')
    ap.add_argument('-p', '--shape_predictor', required=True,
                    help='path to facial landmark predictor')
    arguments = vars(ap.parse_args())

    return arguments


def main(video, predictor, ratio_thresh, frame_thresh):

    # loading the face detector
    print('[INFO] Loading facial landmark predictor...')
    detector = dlib.get_frontal_face_detector()
    # loading the facial landmark predictor
    predictor = dlib.shape_predictor(predictor)

    # saving the landmark indices for the left & the right eye
    left_eye_start, left_eye_end = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    right_eye_start, right_eye_end = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    print('[INFO] Starting video stream...')
    if not video:
        # start web-cam feed
        vs = cv2.VideoCapture(0)
        time.sleep(2.0)

    else:
        # start video stream
        vs = cv2.VideoCapture(args.get('video', False))

    fps = FPS().start()

    counter = 0
    blinks = 0

    # main loop
    while True:

        grabbed, frame = vs.read()

        if frame is None:
            break

        # resize the frame & convert to gray color space
        frame = imutils.resize(frame, width=450)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect all faces in the frame
        rectangles = detector(gray_frame, 0)

        # iterate over al faces
        for face in rectangles:

            # locate all facial landmarks in the frame
            shape = face_utils.shape_to_np(predictor(gray_frame, face))

            # isolate the left & the right eye
            left_eye = shape[left_eye_start:left_eye_end]
            right_eye = shape[right_eye_start:right_eye_end]

            # find the aspect ratio for both the eyes
            left_ratio = eye_aspect_ratio(left_eye)
            right_ratio = eye_aspect_ratio(right_eye)

            # average out the ratio
            avg_ratio = (left_ratio + right_ratio) / 2

            # draw contours around the eyes
            left_eye_curve = cv2.convexHull(left_eye)
            right_eye_curve = cv2.convexHull(right_eye)

            cv2.drawContours(frame, [left_eye_curve], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_curve], -1, (0, 255, 0), 1)

            if avg_ratio < ratio_thresh:
                # increase number of consecutive frames for which the eyes have been closed
                counter += 1

            else:
                if counter >= frame_thresh:
                    # increase number of blinks if the eyes have been closed for more than the threshold number of
                    # frames
                    blinks += 1

                counter = 0

            # label the frame
            cv2.putText(frame, f'Blinks: {blinks}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'iRatio: {round(avg_ratio, 2)}', (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Blink Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        # exit if 'q' is pressed
        if key == ord('q'):
            break

        fps.update()

    fps.stop()
    print(f'[INFO] Elapsed time: {round(fps.elapsed(), 2)}')
    print(f'[INFO] approximate FPS: {round(fps.fps(), 2)}')

    # release endpoint(s) & cleanup
    cv2.destroyAllWindows()
    vs.release()


if __name__ == '__main__':

    args = get_arguments()

    main(video=args.get('video', False), predictor=args['shape_predictor'], ratio_thresh=0.3, frame_thresh=2)
