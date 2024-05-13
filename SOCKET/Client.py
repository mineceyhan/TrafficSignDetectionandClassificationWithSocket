#client.py
from socket import AF_INET, socket, SOCK_STREAM
import struct
from threading import Thread
import os

def server_response():
    while True:
        try:
            msg = client_socket.recv(BUFFERSIZE).decode("utf8")
            print(f"Traffic Sign Class: {msg}")
        except OSError:
            break

def send_image(file_path):
    file_size = os.path.getsize(file_path)
    client_socket.sendall(struct.pack("!I", file_size))  # Dosya boyutunu gönder
    with open(file_path, "rb") as image_file:
        while True:
            image_chunk = image_file.read(BUFFERSIZE)
            if not image_chunk:
                break
            client_socket.send(image_chunk)

HOST = '127.0.0.1'
PORT = 23847
BUFFERSIZE = 1024
ADDR = (HOST, PORT)

client_socket = socket(AF_INET, SOCK_STREAM)
client_socket.connect(ADDR)

file_path = "C:/Users/ceyha/Documents/serverdeneme.jpg"

receive_thread = Thread(target=send_image,args=(file_path,))
receive_thread.start()
server_response()