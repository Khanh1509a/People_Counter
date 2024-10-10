import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture('../Videos/people1.mp4')


model = YOLO("../Yolo-Weights/yolov8n.pt")


classNames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

mask = cv2.imread('Maskpr.png')

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsUp = [1043, 188, 1043, 495]
limitsDown = [110, 480, 110, 700]

totalCountUp = []
totalCountDown = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread("anhNenDem.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics,(0, 0))

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    # detections = np.empty((0, 4))


    for r in results:
        boxes =r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.1:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))


    resultsTracker = tracker.update(detections)

    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        # cvzone.putTextRect(img, f'{classNames[cls]}{conf}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)


        if limitsUp[1] < cy < limitsUp[3] and limitsUp[0] - 15 < cx < limitsUp[0] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[1] < cy < limitsDown[3] and limitsDown[0] - 20 < cx < limitsDown[0] + 20:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)


    cv2.putText(img, str(len(totalCountUp)), (90, 80), cv2.FONT_HERSHEY_PLAIN, 7, (50, 50, 255), 6)
    cv2.putText(img, str(len(totalCountDown)), (280, 80), cv2.FONT_HERSHEY_PLAIN, 7, (50, 50, 255), 6)

    cv2.imshow('Images', img)
    cv2.waitKey(1)
