import socket
import sys

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#get ip address of server
host = socket.gethostname()

#Change port as you desire
port = 1234

server_socket.bind((host, port))

server_socket.listen(10)
print("Server listening. Connect the client to %s" % (host))

while 1:
	client_socket , address = server_socket.accept()
	print("Connection Found. Client IP: %s", address)
	pressed_key = sys.stdin.readline()
	client_socket.send(data.encode())
	print(client_socket.recv(1024).decode())

