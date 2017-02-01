import socket
import sys
import time

#create socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#get local machine name
host = socket.gethostname()

port = 8080

#connection to hostname on port
i = 1
s.connect((host, port))
while i == 1:
	
	#Recieve no more than
	s.send(sys.stdin.readline())
	s.send('\n')
	
	time.sleep(5)
	#print s.recv(1024)
	
s.close

