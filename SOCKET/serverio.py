#server.py 
import base64
import io
import sys
from matplotlib import pyplot as plt
import socketio
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.utils import img_to_array

# Initialize Socket.IO server
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

@sio.event
def connect(sid, environ):
    print('Connected to client', sid)

@sio.event
def disconnect(sid):
    print('Disconnected from client', sid)

# Load cascade classifier and model
cascade_classifier = cv2.CascadeClassifier('cascade.xml')
model = tf.keras.models.load_model('model.h5')

# Define process_image function
@sio.on('image')
def process_image(sid, data):
    # Extract image data from base64 string
    image_data = data['imageData']

    decoded_data = base64.b64decode(image_data)

    # BytesIO kullanarak veriyi okunabilir bir görüntüye dönüştür
    image_stream = io.BytesIO(decoded_data)
    image_data = image_stream.getvalue()

    # 2. Alınan veriyi bir NumPy dizisine dönüştürün
    np_data = np.frombuffer(image_data, dtype=np.uint8)
    print(np_data)
    # 3. NumPy dizisini cv2.imdecode fonksiyonuna geçirin
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    print(img)
    
    if img is None or img.size == 0:
        print('Failed to decode image')
        return
    # Perform image processing
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray_img is None or gray_img.size == 0:
        print('Failed to convert image to grayscale')
        return
    objects = cascade_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in objects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi = img[y:y+h, x:x+w]
        resized_roi = cv2.resize(roi, (30, 30))
        x = img_to_array(resized_roi)
        x /= 255
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images)
        sign_class = np.argmax(classes)
        sign_class = int(sign_class)
        print(sign_class)
        #result back to the client
        sio.emit('result', {'signClass': sign_class}, room=sid)
def start_server():
    import eventlet
    # Start the server
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 23847)), app)
    
if __name__ == '__main__':
    start_server()
