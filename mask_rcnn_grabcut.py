import numpy as np
import argparse
import imutils
import cv2
import os
os.environ['DISPLAY'] = ':0';



ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mask-rcnn", required=True,
	help="base path to mask-rcnn directory")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-u", "--use-gpu", type=bool, default=0,
	help="boolean indicating if CUDA GPU should be used")
ap.add_argument("-e", "--iter", type=int, default=10,
	help="# of GrabCut iterations (larger value => slower runtime)")
args = vars(ap.parse_args())




labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")




weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])


net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)


image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
cv2.imshow("Input", image)


blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
net.setInput(blob)
(boxes, masks) = net.forward(["detection_out_final",
	"detection_masks"])

# loop over the number of detected objects
for i in range(0, boxes.shape[2]):

	classID = int(boxes[0, 0, i, 1])
	confidence = boxes[0, 0, i, 2]
	
	if confidence > args["confidence"]:
		
		(H, W) = image.shape[:2]
		box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
		(startX, startY, endX, endY) = box.astype("int")
		boxW = endX - startX
		boxH = endY - startY

        # extract the pixel-wise segmentation for the object

		mask = masks[i, classID]
		mask = cv2.resize(mask, (boxW, boxH),
			interpolation=cv2.INTER_CUBIC)
		mask = (mask > args["threshold"]).astype("uint8") * 255
		
		
		rcnnMask = np.zeros(image.shape[:2], dtype="uint8")
		rcnnMask[startY:endY, startX:endX] = mask
		# apply a bitwise AND to the input image to show the output
		# of applying the Mask R-CNN mask to the image

		rcnnOutput = cv2.bitwise_and(image, image, mask=rcnnMask)
		

		cv2.imshow("R-CNN Mask", rcnnMask)
		cv2.imshow("R-CNN Output", rcnnOutput)
		
      
		gcMask = rcnnMask.copy()
		gcMask[gcMask > 0] = cv2.GC_PR_FGD
		gcMask[gcMask == 0] = cv2.GC_BGD

		
		print("[INFO] applying GrabCut to '{}' ROI...".format(
			LABELS[classID]))
		fgModel = np.zeros((1, 65), dtype="float")
		bgModel = np.zeros((1, 65), dtype="float")
		(gcMask, bgModel, fgModel) = cv2.grabCut(image, gcMask,
			None, bgModel, fgModel, iterCount=args["iter"],
			mode=cv2.GC_INIT_WITH_MASK)
       
		outputMask = np.where(
			(gcMask == cv2.GC_BGD) | (gcMask == cv2.GC_PR_BGD), 0, 1)
		outputMask = (outputMask * 255).astype("uint8")
		

		output = cv2.bitwise_and(image, image, mask=outputMask)
		
		cv2.imshow("GrabCut Mask", outputMask)
		cv2.imshow("Output", output)
		cv2.waitKey(0)
