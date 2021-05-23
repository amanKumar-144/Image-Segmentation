import numpy as np
import cv2
import tkinter

import os
os.environ['DISPLAY'] = ':0';
  
cv2.imshow("Image",image);


mask = np.zeros(image.shape[:2], np.uint8)
   

backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)
   
rectangle = (20, 20, 340, 120)
   

cv2.grabCut(image, mask, rectangle,  
            backgroundModel, foregroundModel,
            3, cv2.GC_INIT_WITH_RECT)
   

mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
   

image = image * mask2[:, :, np.newaxis]
   
image[np.where((image != [0,0,0]).all(axis = 2))] = [255,255,255]

cv2.imshow("Output",image);
cv2.waitKey(0);