import cv2
from flask import Flask, render_template, Response

print(cv2.__version__)

app = Flask(__name__)
video = cv2.VideoCapture(0)


def sketch(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_img_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)

    canny_edge = cv2.Canny(gray_img_blur, 40, 30)

    ret, mask = cv2.threshold(canny_edge, 150, 255, cv2.THRESH_BINARY_INV)

    ret, buffer = cv2.imencode('.jpg',mask)
    frame = buffer.tobytes()
    return frame


def gen_frames(video):
    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen_frames_sketch(video):
    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + sketch(frame) + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(video), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sketch_feed')
def sketch_feed():
    return Response(gen_frames_sketch(video), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
