import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

_contourScoreZone = np.array([])
subtractorINPUT = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression

yoloINPUT = attempt_load("weights/yolov7-tiny.pt", map_location = torch.device("cuda:0"))


def getAngleList(dirList, pIdxCurr, ptrSuccCurr, ptrXYCurr):
	result = dirList.copy()

	if (ptrSuccCurr == False):
		result.clear()

	if (ptrSuccCurr == True):
		foreSucc = dirList[-2][1] if len(dirList) >= 2 else False
		prevSucc = dirList[-1][1] if len(dirList) >= 1 else False
		currSucc = ptrSuccCurr

		prevX = dirList[-1][2][0] if prevSucc else 0
		prevY = dirList[-1][2][1] if prevSucc else 0
		currX = ptrXYCurr[0]
		currY = ptrXYCurr[1]
		theta = math.degrees(math.atan2((currX - prevX), (currY - prevY)))

		prevAngle = dirList[-1][3] if prevSucc else 0
		currAngle = theta
		delta = abs(currAngle - prevAngle) if currSucc and prevSucc and foreSucc else 0
		direction = 360 - delta if delta > 180 else delta

		result.append((pIdxCurr, ptrSuccCurr, ptrXYCurr, theta, direction))

	else:
		del result[:-1]

	return result



def getLandingList(landingList, angleList):
	'''add to landing list if angle is greater than 45 degrees'''
	result = landingList.copy()

	for angleItem in angleList:
		if (angleItem[4] > 45 and angleItem[2] not in landingList):
			result.append(angleItem[2])

	return result



def getLandingDict(imgOrigin, landingDict, landingList):
    # '''
	# 	check if landing point is in the scorezone, and add to landingDict
	# 	(set the newest)
	# '''
	result = landingDict.copy()

	for landingPtr in result:
		result[landingPtr]["isShuttleNew"] = False

	for landingPtr in landingList:
		if (landingPtr not in landingDict):
			pts = getScoreZone(imgOrigin)			
			ptr = landingPtr
			isShuttleLandIn = bool(cv2.pointPolygonTest(pts, ptr, False) == 1)

			result.update({landingPtr: {
				"isShuttleNew": True,
				"isShuttleLandIn": isShuttleLandIn,
				"contourPolygon": pts
			}})

	return result



def getImgCross(imgSrc, landingDict):
	'''畫在畫面上'''
	result = imgSrc.copy()

	for landingPtr in landingDict.keys():
		leftX = landingPtr[0] - 3
		leftY = landingPtr[1]
		rightX = landingPtr[0] + 3
		rightY = landingPtr[1]
		topX = landingPtr[0]
		topY = landingPtr[1] - 3
		bottomX = landingPtr[0]
		bottomY = landingPtr[1] + 3
		cv2.line(result, (leftX, leftY), (rightX, rightY), (0, 0, 255), 1)
		cv2.line(result, (topX, topY), (bottomX, bottomY), (0, 0, 255), 1)

		isShuttleNew = landingDict[landingPtr]["isShuttleNew"]
		isShuttleLandIn = landingDict[landingPtr]["isShuttleLandIn"]
		contourPolygon = landingDict[landingPtr]["contourPolygon"]

		if True:
			cv2.drawContours(result, [contourPolygon], -1, (0, 255, 0), 2)

		txtLabel = "IN" if isShuttleLandIn else "OUT"
		txtFont = cv2.FONT_HERSHEY_COMPLEX
		cv2.putText(result, txtLabel, landingPtr, txtFont, 0.3, (0, 255, 0))

	return result



def getCenterXY(subtractor, yolo, imgPrev, imgCurr, ptrPrev, tolerance, nameFolder, namePrev, nameCurr):
	binaryThresh = 30

	imgSub = subtractor.apply(imgCurr)
	imgSub[imgSub == 127] = 0

	rectList = []
	device = torch.device("cuda:0")
	conf_thres = 0.25
	iou_thres = 0.45

	for k in ['names', 'stride']:
		setattr(yolo, k, getattr(yolo, k))

	stride = int(yolo.stride.max())
	imgsz = math.ceil(640 / stride) * stride

	## Remove interference by player
	yolo.half()
	names = yolo.names
	im0s = imgCurr
	img = np.ascontiguousarray(im0s[:, :, ::-1].transpose(2, 0, 1))
	img = torch.from_numpy(img).to(device)
	img = img.half()
	img /= 255.0

	if img.ndimension() == 3:
		img = img.unsqueeze(0)

	with torch.no_grad():
		pred = yolo(img, augment = False)[0]

	pred = non_max_suppression(pred, conf_thres, iou_thres, classes = None, agnostic = False)

	for i, det in enumerate(pred):
		im0 = im0s.copy()

		if len(det):
			for *xyxy, conf, cls in reversed(det):
				boxName = names[int(cls)]
				boxConf = conf.item()
				boxX1 = int(xyxy[0].item())
				boxY1 = int(xyxy[1].item())
				boxX2 = int(xyxy[2].item())
				boxY2 = int(xyxy[3].item())

				if (boxName == "person"):
					pt1 = (boxX1, boxY1)
					pt2 = (boxX2, boxY2)
					rectList.append((pt1, pt2))
					im0 = cv2.rectangle(im0, (boxX1,boxY1), (boxX2,boxY2), (0,255,0), 6)

	imgHumanArea = np.zeros(imgSub.shape, dtype=np.uint8)

	for rects in rectList:		
		imgHumanArea = cv2.rectangle(imgHumanArea, rects[0], (rects[1][0], rects[1][1] * 2), 255, -1)

	imgNoHumanMask = cv2.bitwise_not(imgHumanArea)
	kernel = np.ones((3, 3), np.uint8)
	imgNoHumanMask = cv2.erode(imgNoHumanMask, kernel, iterations = 2)
	imgNoHuman = cv2.bitwise_and(imgSub, imgSub, mask = imgNoHumanMask)
	retv, thresh = cv2.threshold(imgNoHuman, binaryThresh, 255, cv2.THRESH_BINARY)

	threshInv = cv2.bitwise_not(thresh)
	kernel = np.ones((3, 3), np.uint8)
	erosionInv = cv2.erode(threshInv, kernel, iterations = 1)
	erosion = cv2.bitwise_not(erosionInv)

	contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	imgContour = imgCurr.copy()

	result = (False, ptrPrev, imgCurr)
	ptList = []

	for ctr in contours:
		mObj = cv2.moments(ctr)
		if mObj['m00'] != 0:
			ctrCntX = int(mObj['m10']/mObj['m00'])
			ctrCntY = int(mObj['m01']/mObj['m00'])
			ctrArea = cv2.contourArea(ctr)
			ctrRectX, ctrRectY, ctrRectW, ctrRectH = cv2.boundingRect(ctr)
			ctrRectRatio = ctrRectW / ctrRectH

			if (ctrArea >= 200 and ctrArea <= 1000 and ctrRectRatio >= 0.5 and ctrRectRatio <= 2):
				ptList.append((ctrCntX, ctrCntY))
				cv2.drawContours(imgContour, [ctr], -1, (0, 0, 255), 1)

	result = (True, ptList[0], imgContour) if len(ptList) == 1 else (False, ptrPrev, imgCurr)

	return result



def getScoreZoneInitial(img:np.ndarray) -> np.ndarray:
	result = [0, 5, 298, 12, 614, 450, 0, 462]

	try:
		f = open("../court.txt", "r")
		strXY = f.read()
		f.close()

		XYs = strXY.split(", ")
		result = list(map(lambda pos: int(pos), XYs))

	except ValueError as e:
		print(e)

	except:
		print("something wrong")

	return np.array(result).reshape(-1, 1, 2)



def getScoreZone(img:np.ndarray) -> np.ndarray:
	return _contourScoreZone



inputVideoName = sys.argv[1]
print(inputVideoName)
cap = cv2.VideoCapture(inputVideoName)

fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
#print(fourcc)
fps = cap.get(cv2.CAP_PROP_FPS)

outputWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
outputHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
outputVideo = cv2.VideoWriter(inputVideoName[:-4] + "_predict" + inputVideoName[-4:], fourcc, fps, (outputWidth, outputHeight))
#outputVideo = cv2.VideoWriter("../video/0.mp4", fourcc, fps, (outputWidth, outputHeight))
prevImg = np.zeros((outputHeight, outputWidth, 3), np.uint8)

frIdx = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, frIdx)
success, frame = cap.read()

landingList = []
landingDict = {}
directionList = []
trackingList = []
ptrPrevINPUT = (0, 0)
toleranceINPUT = 0
_contourScoreZone = getScoreZoneInitial(frame)
while success:
	nameFolderINPUT = "imgLND"
	namePrevINPUT = "%03d" %(frIdx - 1)
	nameCurrINPUT = "%03d" %(frIdx)
	imgPrevINPUT = prevImg.copy()
	imgCurrINPUT = frame.copy()

	retSuccessful, ptrCurrOUTPUT, imgResultOUTPUT = getCenterXY(subtractorINPUT, yoloINPUT, imgPrevINPUT, imgCurrINPUT, ptrPrevINPUT, toleranceINPUT, nameFolderINPUT, namePrevINPUT, nameCurrINPUT)
	toleranceINPUT = 0 if retSuccessful else min(3, toleranceINPUT + 1)
	print("{0} --> {1}".format(nameCurrINPUT, ptrCurrOUTPUT))
	ptrPrevINPUT = ptrCurrOUTPUT
	trackingList.append(ptrPrevINPUT)
	if (retSuccessful):
		directionList = getAngleList(directionList, frIdx, retSuccessful, ptrCurrOUTPUT)
		landingList = getLandingList(landingList, directionList)
		landingDict = getLandingDict(frame, landingDict, landingList)
		imgWithLanding = getImgCross(imgResultOUTPUT, landingDict)
		outputVideo.write(imgWithLanding)
		#cv2.imwrite("../images/1.png", imgWithLanding)

	else:
		directionList = getAngleList(directionList, frIdx, False, (0, 0))
		landingList = getLandingList(landingList, directionList)
		landingDict = getLandingDict(frame, landingDict, landingList)
		imgWithLanding = getImgCross(frame, landingDict)
		outputVideo.write(imgWithLanding)
		#cv2.imwrite("../images/1.png", imgWithLanding)

	success, frame = cap.read()
	frIdx += 1




