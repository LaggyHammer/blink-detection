# blink-detection
## dlib-blink-detection
Blink detection using facial landmark recognition.

Using dlib's face detector, faces are detected in each frame of a video stream.
Iterating over each detected face, we predict the facial landmarks using a pre-trained predictor.

The landmarks for each eye are isolated & an average aspect ratio for the eyes is calculated.
If this ratio falls below a certain customizable threshold for _'n'_ consecutive frames,
the blink counter is increased.

## Usage
```commandline
python app.py
python dlib_blink_detection.py --shape_predictor shape_predictor_68_face_landmarks.dat
python dlib_blink_detection.py --video test.mp4 --shape_predictor shape_predictor_68_face_landmarks.dat
```
