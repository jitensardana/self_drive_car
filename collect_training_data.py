import pygame
import serial
import cv2
import numpy as np
import sys
from AndroidCamFeed import AndroidCamFeed

class CollectTrainingData:
    def __init__(self):
        self.host = sys.argv[1]
        self.connected = False

        ## Pygame stuff
        pygame.init()
        size = (320, 240)
        pygame.display.set_mode(size)

        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1

        '''
        # Connect to serial port for arduino
        self.ser = serial.Serial('/dev/tty.usbmodem1421', 115200, timeout=1)
        '''

        self.send_inst = True
        self.collect_image()

    def collect_image(self):
        saved_frames = 0
        total_frames = 0

        acf = AndroidCamFeed(self.host)
        self.connected = True
        cv2.namedWindow('Android Feed', cv2.WINDOW_AUTOSIZE)

        print "Start collecting images .. \n"

        number_pic = 0

        image_array = np.zeros((1, 38400))
        label_array = np.zeros((1, 4), 'float')

        if not self.connected:
            print "Error in connection"
            return
        else:
            try:
                while self.send_inst and acf.isOpened():
                    ret, frame = acf.read()
                    if ret:
                        cv2.imshow('Android Feed', frame)
                        cv2.waitKey(1)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        image = cv2.GaussianBlur(frame, (5, 5), 0)
                        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        #image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        file_name = 'test' + str(number_pic) + '.jpg'
                        cv2.imwrite('testing_data/'+file_name, image)

                        roi = image[120:240, :]
                        # image_temp = np.asarray(image)  # copy image so that original does not get destroyed

                        total_frames += 1

                        temp_array = roi.reshape(1, 38400).astype(np.float32)

                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            key_input = pygame.key.get_pressed()
                            number_pic += 1
                            filename = 'test' + str(number_pic) + '.jpg'
                            cv2.imwrite('testing_images/' + filename, frame)

                            # complex orders
                            if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                                print "Forward Right"
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[1]))
                                saved_frames += 1
                            # self.ser.write(chr(6))
                            elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                                print "Forward Left"
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[0]))
                                saved_frames += 1
                            # self.ser.write(chr(7))
                            elif key_input[pygame.K_DOWN] and key_input[pygame.K_RIGHT]:
                                print "Reverse Right"
                            # self.ser.write(chr(8))
                            elif key_input[pygame.K_DOWN] and key_input[pygame.K_LEFT]:
                                print "Reverse Left"
                            # self.ser.write(chr(9))

                            # simple orders
                            elif key_input[pygame.K_UP]:
                                print "Forward"
                                saved_frames += 1
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[2]))
                            # self.ser.write(chr(1))
                            elif key_input[pygame.K_DOWN]:
                                print "Reverse"
                                saved_frames += 1
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[3]))
                            # self.ser.write(chr(2))
                            elif key_input[pygame.K_RIGHT]:
                                print "Right"
                                saved_frames += 1
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[1]))
                            # self.ser.write(chr(3))
                            elif key_input[pygame.K_LEFT]:
                                print "Left"
                                saved_frames += 1
                                image_array = np.vstack((image_array, temp_array))
                                label_array = np.vstack((label_array, self.k[0]))
                            # self.ser.write(chr(4))
                            elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                                print 'exit'
                                self.send_inst = False
                                # self.ser.write(chr(0))
                                break
                        elif event.type == pygame.KEYUP:
                            pass  # self.ser.write(chr(0))

                '''
                Save training images and labels
                '''
                train = image_array[1:, :]
                train_labels = label_array[1:, :]
                np.savez('testing_data/testing.npz', train=train, train_labels=train_labels)
                print "Collection Completed \n"
                print(train.shape)
                print(train_labels.shape)
                print "Total frames : ", total_frames
                print "Saved frames : ", saved_frames
                print "Dropped frames : ", total_frames - saved_frames

            finally:
                pygame.quit()
                acf.release()
                cv2.destroyAllWindows()


if __name__ == '__main__':
    CollectTrainingData()
