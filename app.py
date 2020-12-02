from flask import Flask, Response
import cv2
from dlib_blink_detection import web_main

app = Flask(__name__)
video = cv2.VideoCapture(0)


@app.route('/')
def index():
    return "Default Message"


def gen(video):
    while True:
        success, image = video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    global video
    return Response(web_main(video, 'shape_predictor_68_face_landmarks.dat', ratio_thresh=0.3, frame_thresh=2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
