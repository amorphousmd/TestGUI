import socket
import time

x = [(401, 518)]

def createData(coords_list):
    list1 = [elem for coords in coords_list for elem in coords]
    msg = ""
    for i in list1:
        msg += ","
        msg += str(i)
    msg = str(len(coords_list)) + msg
    return msg

def zeroExtend(inputList):
    output = []
    for tup in inputList:
        tup = (*tup, 0)
        output.append(tup)
    return output

HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
SERVER = "127.0.0.1"
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(send_length)
    client.send(message)
    print(client.recv(2048).decode(FORMAT))

send(createData(zeroExtend(x)))
time.sleep(1)
send(createData(zeroExtend(x)))
time.sleep(1)
send(createData(zeroExtend(x)))
