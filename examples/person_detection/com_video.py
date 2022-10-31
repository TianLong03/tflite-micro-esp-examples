import socket
import sys
import cv2 as cv
import numpy as np

from threading import Thread
import time
import tensorflow as tf


PORT = 8088
INTERFACE = 'eth0'

train_list = ['notperson','personStanding','personSitting','personProne']

color_list = [
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255),
    (100, 100, 0),
    (0, 100, 100),
    (100, 0, 100),
    (100, 0, 0),
]

canvas = np.zeros((960, 960, 3))
cv.imshow("person_detection", canvas)
cv.waitKey(100)

recv_data = np.zeros((96*96,), np.uint8)
# feature_data = np.zeros((96*96,), np.int8)

tflite_model_file = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

class RecvThread(Thread):
    def __init__(self, sock, payload):
        super().__init__()
        self.counter = 0
        self.running = True
        self.sock = sock
        self.payload = payload
    
    def run(self):
        
        started = False

        while(1):
            recv_len = 0
            data = None
            while recv_len < 96*96:
                try:
                    data = self.sock.recv(100*100)
                except:
                    print("sock error")
                    break
                
                arr = np.frombuffer(data, dtype=np.int8)
                print("arr.shape=", arr.shape)
                # feature_data[recv_len:recv_len+arr.shape[0]] = arr
                recv_data[recv_len:recv_len+arr.shape[0]] = arr + 128
            
                recv_len += arr.shape[0]

            if data is None:
                print("not data")
                break

            
            
            self.sock.sendall(self.payload.encode())
            canvas = recv_data.reshape((96, 96, 1))
            canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)
            cv.imwrite(f"logs/img_{time.strftime('%Y%m%d_%H%M%S')}.png", canvas)

            

        self.running = False

def tcp_client(address, payload):
    global recv_data
    # global feature_data
    
    cur_pred_pos = 0

    
    started = False

    t = None
    for res in socket.getaddrinfo(address, PORT, socket.AF_UNSPEC,
                                  socket.SOCK_STREAM, 0, socket.AI_PASSIVE):
        family_addr, socktype, proto, canonname, addr = res
    try:
        sock = socket.socket(family_addr, socket.SOCK_STREAM)
        sock.settimeout(60.0)
    except socket.error as msg:
        print('Could not create socket: ' + str(msg[0]) + ': ' + msg[1])
        return
    try:
        sock.connect(addr)
    except socket.error as msg:
        print('Could not open socket: ', msg)
        sock.close()
        return
    
    while(1):
        if started == False:
            print("create thread")
            started = True
            t = RecvThread(sock, payload)
            t.start()

        

        canvas = recv_data.reshape((96, 96, 1))

        feature = np.expand_dims(canvas, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], feature)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        
        print(f"output:", output)
        # print(recv_data.shape)

        canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)
        canvas = cv.resize(canvas, (640, 640))

        for j in range(len(train_list)):
            cv.putText(canvas, f"{train_list[j]}({output[j]+128})", (10, 100*j+40), cv.FONT_HERSHEY_SIMPLEX, 1, color_list[j], 1, cv.LINE_AA)
            cv.rectangle(canvas, (10, 100*j+50), ((output[j]+256), 100*j+90), color_list[j], -1)
        # cv.imshow("predictions", canvas)

        # print(canvas.shape)
        cv.imshow("person_detection", canvas)
        cv.waitKey(1)
        if not t.running:
            break    
        
if __name__ == '__main__':
    if sys.argv[2:]:    # if two arguments provided:
        # Usage: example_test.py <server_address> <message_to_send_to_server>
        tcp_client(sys.argv[1], sys.argv[2])
    # else:               # otherwise run standard example test as in the CI
    #     test_examples_protocol_socket_tcpserver()