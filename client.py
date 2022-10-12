from multiprocessing import Process
from multiprocessing import Pipe
import socket

clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
conn1, conn2 = Pipe(duplex=True)


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


def startClient():
    receiver_process = Process(target=receiver, args=(conn1,))
    receiver_process.start()


def sendData(data):
    sender_process = Process(target=sender, args=(conn2, createData(data)))
    sender_process.start()


def stopClient():  # Stopping the client also stops the server
    sender_process = Process(target=sender, args=(conn2, 'STOP'))
    sender_process.start()


def sender(connection, data):
    print('Getting Data', flush=True)
    connection.send(data)
    # all done
    print('Data Sent', flush=True)


def receiver(connection):
    # HOST = '192.168.1.1'
    HOST = '127.0.0.1'
    PORT = 5050
    server_address = (HOST, PORT)
    print(f'[CONNECTING] TO PORT: {server_address}')
    try:
        clientSocket.connect(server_address)
    except ConnectionRefusedError:
        print("[ERROR] Server not found. Make sure the server is opened")
    else:
        while True:
            data = connection.recv()
            if data == 'STOP':  # Data received = 'STOP' -> Close the client and the server
                clientSocket.close()
                break
            else:
                clientSocket.sendall(bytes(data, "utf8"))
