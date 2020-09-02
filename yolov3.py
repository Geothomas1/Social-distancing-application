import os
import time
import itertools
import cv2
import numpy as np


class PeopleDetector:
    def __init__(self, yolocfg='yolo_weights/yolov3.cfg',
                 yoloweights='yolo_weights/yolov3.weights'):
        self.net = None
        self.__yolocfg = yolocfg
        self.__yoloweights = yoloweights
        self.__layer_names = None
        self.__layerouts = []

    def load_network(self):
        self.net = cv2.dnn.readNetFromDarknet(
            self.__yolocfg, self.__yoloweights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.__layer_names = [self.net.getLayerNames()[i[0] - 1]
                              for i in self.net.getUnconnectedOutLayers()]
        print("yolov3 loaded successfully\n")

    def predict(self, image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        start = time.time()
        self.__layerouts = self.net.forward(self.__layer_names)
        end = time.time()
        #print("yolo took {:.6f} seconds".format(end - start))
        return(self.__layerouts)

    def clear_outs(self):
        self.__layerouts = []

    # def process_preds(self, image, outs):
    #     (frameHeight, frameWidth) = image.shape[:2]
    #     for out in outs:
    #         for detection in out:
    #             scores = detection[5:]
    #             classId = np.argmax(scores)
    #             if classId != 0:  # filter person class
    #                 continue
    #             confidence = scores[classId]
    #             if confidence > self.__confidence:
    #                 center_x = int(detection[0] * frameWidth)
    #                 center_y = int(detection[1] * frameHeight)
    #                 width = int(detection[2] * frameWidth)
    #                 height = int(detection[3] * frameHeight)
    #                 left = int(center_x - width / 2)
    #                 top = int(center_y - height / 2)
    #                 self.__classIDs.append(classId)
    #                 self.__confidences.append(float(confidence))
    #                 self.__boxes.append([left, top, width, height])
    #                 self.__centers.append((center_x, center_y))
    #     indices = cv2.dnn.NMSBoxes(
    #         self.__boxes, self.__confidences, self.__confidence, self.__nmsthreshold)
    #     for i in indices:
    #         i = i[0]
    #         box = self.__boxes[i]
    #         left = box[0]
    #         top = box[1]
    #         width = box[2]
    #         height = box[3]
    #         self.draw_pred(image, self.__classIDs[i], self.__confidences[i], left,
    #                        top, left + width, top + height)
    #     return self.__centers

    # def clear_preds(self):
    #     self.__boxes = []
    #     self.__confidences = []
    #     self.__classIDs = []
    #     self.__centers = []
    #     self.__layerouts = []
    #     self.__mindistances = {}

    # def draw_pred(self, frame, classId, conf, left, top, right, bottom):
    #     cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    #     label = '%.2f' % conf
    #     label = '%s:%s' % (self.__labels[classId], label)
    #     labelSize, baseLine = cv2.getTextSize(
    #         label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #     top = max(top, labelSize[1])
    #     cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(
    #         1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    #     cv2.putText(frame, label, (left, top),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    #     self.find_min_distance(self.__centers)
    #     for k in self.__mindistances:
    #         cv2.line(frame, k[0], k[1], (0, 0, 255), 7)

    # def find_min_distance(self, centers):
    #     '''
    #     return min euclidean distance between predicted anchor boxes
    #     '''
    #     centers = self.__centers
    #     comp = list(itertools.combinations(centers, 2))
    #     for pts in comp:
    #         ecdist = np.linalg.norm(np.asarray(pts[0])-np.asarray(pts[1]))
    #         if ecdist < self.__MIN_DIST:
    #             self.__mindistances.update({pts: ecdist})
