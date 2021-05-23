import cv2
import numpy as np

import os
os.environ['DISPLAY'] = ':0';
 
image = cv2.imread("Image3_seg.png")
# Loading the image
 
image= cv2.resize(image, (400, 267));
cv2.imshow("Image",image);
cv2.waitKey(0);

