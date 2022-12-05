import pickle
from flask_socketio import SocketIO
import cv2
from flask import Flask, render_template, Response

"""This file liveVideo deals with generating the frames (video feed) for the livestream that will be viewable on the 
index page. It also loads the cascade face recognition model for face detection and recognition. It then draws a 
rectangle around the detected face and labels it with the corresponding name """

app = Flask(__name__)
socketioApp = SocketIO(app)
camera = cv2.VideoCapture(0)

"""This function generates frames for livestream and loads facial recognition model previously trained"""
def gen_frames():
    # face cascade model loaded
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
    # creates recognizer and reads trainer.yml
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    # loads label file created from directory names using pickle
    with open('pickles/labels.pickle', 'rb') as f:
        og_labels = pickle.load(f)
        # reverse the og_labels dictionary
        labels = {v: k for k, v in og_labels.items()}

    while True:
        frame_counter = int(camera.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = camera.read()
        if (frame_counter % 2 == 0):  # mod by 2 for every second frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# grayscale conversion
            # check if face exists in image frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
            for (x, y, w, h) in faces:
                # get region of interest
                roi_gray = gray[y:y + h, x:x + w]


                # use the region of interest to detect known face in frame
                id_, conf = recognizer.predict(roi_gray)
                # confidence interval for detection
                if conf >= 30 and conf <= 85:
                    # add label to frame
                    cv2.putText(frame,
                                labels[id_] + " Match: " + str(round(conf, 2)) + "%",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA)

                color = (255, 0, 0)  # Blue
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                # draw rectangle around known face
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

                break

            # Display the resulting frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        # else return frame with no processing
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        # break on usr pressed q
        if cv2.waitKey(1) == ord('q'):  #
            break

    camera.release()
    cv2.destroyAllWindows()


# renders the index.html page found in templates directory
@app.route('/')
def index():
    return render_template('index.html')


# sends frames to the user that is within the index.html
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run():
    socketioApp.run(app)


if __name__ == '__main__':
    socketioApp.run(app)
