__author__ = 'jitensardana'

import pygame
import cv2
import numpy as np
import socket
from pygame.locals import *
import sys
from AndroidCamFeed import AndroidCamFeed

class CollectTainingData(object):
	def __init__(self):
		self.host = sys.argv[1]
		self.connected = False
		pygame.init()
		self.collect_image(self.host)

		#labels
		self.k = np.zeros((4,4), 'float')
		for i in range(4):
			self.k[i,i] = 1
		self.temp_label = np.zeros((1,4), 'float') 	

		# connect to a serial port for arduino
        self.ser = serial.Serial('/dev/tty.usbmodem1421', 115200, timeout=1)
        self.send_inst = True

	def collect_image(self, host):
		
		saved_frame = 0
		total_frames = 0

		acf = AndroidCamFeed(host)
		self.connected = True
		cv2.namedWindow('feed', cv2.WINDOW_NORMAL)

		print "Start Collecting images ..."

		no_pic = 0  #number of pictures

		image_array = np.zeros((1,38400))
		label_array = np.zeros((1,4), 'float')	
		

		if self.connected == False:
			print "Error in connection"
			return
		else:
			try:
				while self.send_inst and acf.isOpened():
					

					ret, frame = acf.read()
					
            		if ret :
            			cv2.imshow('feed', frame)
            			image = cv2.GaussianBlur(image,(5,5),0)	
            			image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            			image = cv2.threshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)	
            		
            		cv2.imwrite("train" + no_pic , image)

            		image_temp= image.copy() #coying image so that original does not get destroyed

            		total_frames += 1
            		no_pic += 1

            		temp_array = image_temp.reshape(1,38400).astype(np.float32)

            		for event in pygame.event.get():
            			if event.type = KEYDOWN:
            				key_input = pygame.key.get_pressed()

            				#complex orders
            				if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
            					print "Forward Right"
            					image_array = np.vstack((image_array, temp_array))
            					label_array = np.vstack((label_array, self.k[1]))
            					saved_frame += 1
            					self.ser.write(chr(6))
            				elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
            					print "Forward Left"
            					image_array = np.vstack((image_array, temp_array))
            					label_array = np.vstack((label_array, self.k[0]))
            					saved_frame += 1
            					self.ser.write(chr(7))
            				elif key_input[pygame.K_DOWN] and key_input[pygame.K_RIGHT]:
            					print "Reverse Right"
            					self.ser.write(chr(8))
            				elif key_input[pygame.K_DOWN] and key_input[pygame.K_LEFT]:
            					print "Reverse Left"
            					self.ser.write(chr(9))

            				#simple orders
            				elif key_input[pygame.K_UP]:
            					print "Forward"
            					saved_frame += 1
            					image_array = np.vstack((image_array, temp_array))
            					label_array = np.vstack((label_array, self.k[2]))
            					self.ser.write(chr(1))
            				elif key_input[pygame.K_DOWN]:
            					print "Reverse"
            					saved_frame += 1
            					image_array = np.vstack((image_array, temp_array))
            					label_array = np.vstack((label_array, self.k[3]))
            					self.ser.write(chr(2))
            				elif key_input[pygame.K_RIGHT]:
            					print "Right"
            					saved_frame += 1
            					image_array = np.vstack((image_array, temp_array))
            					label_array = np.vstack((label_array, self.k[1]))
            					self.ser.write(chr(3))
            				elif key_input[pygame.K_LEFT]:
            					print "Left"
            					saved_frame += 1
            					image_array = np.vstack((image_array, temp_array))
            					label_array = np.vstack((label_array, self.k[0]))
            					self.ser.write(chr(4))
            				elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                                print 'exit'
                                self.send_inst = False
                                self.ser.write(chr(0))
                                break

                        elif event.type == pygame.KEYUP:
                            self.ser.write(chr(0))

                # save training images and labels

                train = immage_array[1:, :]
                train_labels = label_array[1:, :]

                np.savez('training_data/training.npz', train = train, train_labels = train_labels)

                print "Training Completed \n"

                print(train.shape)
                print(train_labels.shape)

                print "Total frames : ", total_frames
                print "Saved frames : ", saved_frame
                print "Dropped frames : ", total_frames - saved_frame

            finally:
            	acf.release()
            	cv2.destroyAllWindows()


if __name__ == '__main__':
	CollectTrainingData()

            	




						






