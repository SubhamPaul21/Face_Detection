from imutils.video import VideoStream
from facedetector import FaceDetector
from flask import Response, Flask, render_template
import cv2
import imutils
import threading
import time
import argparse


# Load haar cascade files
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__, template_folder='templates')
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
    return render_template("index.html")

def detect_motion(frameCount):
    global vs, outputFrame, lock
    total = 0
    
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        output = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if total > frameCount:
            output = FaceDetector.detect(gray, frame)
        
        total += 1
        with lock:
            outputFrame = output.copy()
            
def generate():
    # grab global references to the output frame and lock variables
	global outputFrame, lock

    # loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
        bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
                 mimetype = "multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, default="127.0.0.1",
                 help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, default=5000,
                 help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
                 help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
         threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()
