import cv2
import numpy as np
import os
import copy


def nothing(x):
    pass


image_dir = '/home/dung/Project/AI/DeepFashion_Try_On/Data_preprocessing/ACGPN_traindata/train_color'
edge_dir = '/home/dung/Project/AI/DeepFashion_Try_On/Data_preprocessing/ACGPN_traindata/train_edge'
folder = os.listdir(image_dir)
cv2.namedWindow('image')
# create trackbars for color change
cv2.createTrackbar('lowH', 'image', 0, 179, nothing)
cv2.createTrackbar('highH', 'image', 179, 179, nothing)

cv2.createTrackbar('lowS', 'image', 0, 255, nothing)
cv2.createTrackbar('highS', 'image', 255, 255, nothing)

cv2.createTrackbar('lowV', 'image', 0, 255, nothing)
cv2.createTrackbar('highV', 'image', 255, 255, nothing)

for i, f in enumerate(folder):
    print('image : ', f)
    frame = cv2.imread('{}/{}'.format(image_dir, f), cv2.IMREAD_COLOR)
    origin = copy.copy(frame)
    edge = cv2.imread('{}/{}'.format(edge_dir, f), cv2.IMREAD_COLOR)
    while(1):

        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame = origin
        # get current positions of the trackbars
        ilowH = cv2.getTrackbarPos('lowH', 'image')
        ihighH = cv2.getTrackbarPos('highH', 'image')
        ilowS = cv2.getTrackbarPos('lowS', 'image')
        ihighS = cv2.getTrackbarPos('highS', 'image')
        ilowV = cv2.getTrackbarPos('lowV', 'image')
        ihighV = cv2.getTrackbarPos('highV', 'image')

        # convert color to hsv because it is easy to track colors in this color model
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([ilowH, ilowS, ilowV])
        higher_hsv = np.array([ihighH, ihighS, ihighV])
        # Apply the cv2.inrange method to create a mask
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
        # Apply the mask on the image to extract the original color
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        frame = np.concatenate((frame, origin, edge), axis=1)
