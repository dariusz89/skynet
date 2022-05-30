from gevent import monkey
monkey.patch_all()

from importlib import import_module
import os, io
from flask import Flask, render_template, Response

from flask import send_from_directory
from flask_socketio import SocketIO, emit
import numpy as np
from PIL import Image
import cv2
import json
from network.skynet import net

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)
app.config['SECRET_KEY'] = 'm3J4=u$XUrm2A<sw#GGDffPD5sF86vARXWx6R@+qssQ^$H@^3GJG@bY64hC<dR3ArZkHZs_RkZWomWsMhA8jq/NSs555y_jqb4_AVG6Em^^73xxK<5*@eTT7@#Bz<xpn'
socketio = SocketIO(app,async_mode="gevent")

@socketio.on('current frame')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    lists = convertToNumpiArray(Camera()).tolist()
    ans,out = net(np.reshape(lists, -1))
    json_str = {"ans":str(ans),"out":out.tolist(),"data":  lists}
    socketio.emit('new frame', json_str)
    #print('OK')

@app.route('/')
def index():
    """Video streaming home page."""
    title='Skynet Baby'
    return render_template('index.html', the_title=title)

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

def gen(camera):
    """Video streaming generator function."""
    yield b'--frame\r\n'
    while True:
        frame = camera.get_frame()
        #convertToNumpiArray(Camera())
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

def convertToNumpiArray(camera):
    frame = camera.get_frame()
    im = np.frombuffer(frame, dtype=np.uint8)
    frameNpArray = cv2.imdecode(im,3)
    frameNpArray = cv2.cvtColor(frameNpArray, cv2.COLOR_BGR2GRAY) #Convert to gray
    frameNpArray = cv2.GaussianBlur(frameNpArray, (7, 7), 0) #blurred
    (T, threshInv) = cv2.threshold(frameNpArray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    frameNpArray = threshInv
    scale_percent = 10 # percent of original size
    width = int(frameNpArray.shape[1] * scale_percent / 100)
    height = int(frameNpArray.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(frameNpArray, dim, interpolation = cv2.INTER_AREA)
    resized = (resized > 127)
    return resized

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', debug=True)
