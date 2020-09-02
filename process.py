import numpy as np
import cv2


class PostProcessor:
    def __init__(self, filterlist=[0], confidence=0.8,
                 nmsthreshold=0.4):
        self.__boxes = []
        self.__confidences = []
        self.__classIDs = []
        self.__centers = []
        self.__filterlist = filterlist  # 0 >> filtering persons
        self.__confidence = confidence
        self.__nmsthreshold = nmsthreshold
        self.count = 0


    def process_preds(self, frame, outs):
        (frameHeight, frameWidth) = frame.shape[:2]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                
                if classId not in self.__filterlist:
                    continue
                confidence = scores[classId]
                if confidence > self.__confidence:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    self.count+=1
                    self.__classIDs.append(classId)
                    self.__confidences.append(float(confidence))
                    self.__boxes.append([left, top, width, height])
                    self.__centers.append((center_x, center_y))
        indices = cv2.dnn.NMSBoxes(
            self.__boxes, self.__confidences, self.__confidence, self.__nmsthreshold)
        return indices, self.__boxes, self.__classIDs, self.__confidences, self.__centers,self.count
       
        
