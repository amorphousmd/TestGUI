import socket

HOST = '127.0.0.1'
# HOST = socket.gethostbyname(socket.gethostname())
PORT = 5050

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(2)
print(f"[LISTENING] Server is listening on {HOST}")

while True:
    client, addr = s.accept()

    try:
        print('[CONNECTED] by', addr)
        while True:
            data = client.recv(1024)
            str_data = data.decode("utf8")
            if str_data == "" or str_data == "quit":
                break
            print("Client: " + str_data)

    finally:
        s.close()
        break
