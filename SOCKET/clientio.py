#client.py
import socketio
import os

sio = socketio.Client()

@sio.event
def connect():
    print('Connected to server')

@sio.event
def connect_error():
    print('Connection failed')

@sio.event
def disconnect():
    print('Disconnected from server')
    
@sio.event
def result(data):
    print('Received result from server:', data)
    
def send_image(file_path):
    file_size = os.path.getsize(file_path)
    with open(file_path, "rb") as image_file:
        image_data = image_file.read()
    print("gönderiliyor")
    sio.emit('image', {'fileSize': file_size, 'imageData': image_data})

if __name__ == '__main__':
    sio.connect('http://127.0.0.1:23847')

    file_path = "C:/Users/ceyha/Documents/serverdeneme.jpg"
    send_image(file_path)
    
    # sunucudan gelen verileri beklemek için kullanılır
    sio.wait()
    sio.disconnect()
