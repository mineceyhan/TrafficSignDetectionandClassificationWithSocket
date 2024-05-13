#server.py 
from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread
import os
import cv2
import struct
import numpy as np
import tensorflow as tf
import numpy as np
from keras.utils import img_to_array

current_dir = os.path.dirname(os.path.abspath(__file__))
cascade_xml_path = os.path.join(current_dir, "cascade.xml")
model_h5_path = os.path.join(current_dir, "model.h5")

cascade_classifier = cv2.CascadeClassifier(cascade_xml_path)
model = tf.keras.models.load_model(model_h5_path)

addresses = {}

def process_image(client_socket):
    """Görüntü işleme fonksiyonu"""
    file_size_data = client_socket.recv(4)  # Dosya boyutunu al
    file_size = struct.unpack("!I", file_size_data)[0]  # Baytları dosya boyutuna çöz
    received_size = 0
    received_data = b""  # Alınan veriyi biriktireceğimiz boş bir bayt dizisi
    while received_size < file_size:
        image_chunk = client_socket.recv(BUFFERSIZE)
        received_data += image_chunk
        received_size += len(image_chunk)
    
    img = cv2.imdecode(np.frombuffer(received_data, np.uint8), cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        client_socket.send(bytes(str(sign_class), "utf8"))
        
HOST = '127.0.0.1'
PORT = 23847
BUFFERSIZE = 1024
ADDR = (HOST, PORT)
SERVER = socket(AF_INET, SOCK_STREAM)
SERVER.bind(ADDR)

if __name__ == "__main__":
    SERVER.listen(10)
    print("Connection Waiting...")
    while True:
        client_socket, client_address = SERVER.accept()
        print("%s:%s connected." % client_address)
        addresses[client_socket] = client_address

        ACCEPT_THREAD = Thread(target=process_image, args=(client_socket,))
        ACCEPT_THREAD.start()
