from utils.utils import find_min_distance
import cv2
from pygame import mixer 

class Visualizer:
    def __init__(self, labelpath='yolo_weights/coco.names', detected_object_rect_color=(255, 178, 50),
                 detected_object_rect_thickness=3, label_font=cv2.FONT_HERSHEY_SIMPLEX, label_fontscale=0.5,
                 label_font_thickness=1, label_rect_color=(255, 255, 255), label_text_color=(0, 0, 0),
                 critical_line_color=(0, 0, 255), critical_line_thickness=7):
        self._labels = open(labelpath).read().strip().split("\n")
        self.detected_object_rect_color = detected_object_rect_color
        self.detected_object_rect_thickness = detected_object_rect_thickness
        self.label_font = label_font
        self.label_fontscale = label_fontscale
        self.label_font_thickness = label_font_thickness
        self.label_rect_color = label_rect_color
        self.label_text_color = label_text_color
        self.critical_line_color = critical_line_color
        self.critical_line_thickness = critical_line_thickness


class video(Visualizer):
    def __init__(self, nmsboxes, frame, classIds, confs, boxes, centers,count):
        super().__init__()
        self.__nmsboxes = nmsboxes
        self.__frame = frame
        self.__boxes = boxes
        self.__classIds = classIds
        self.__confs = confs
        self.__centers = centers
        self.__critical_dists = {}
        
    def draw_pred(self):
        for i in self.__nmsboxes:
            i = i[0]
           
            box = self.__boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv2.rectangle(self.__frame, (left, top),
                          (left+width, top+height), self.detected_object_rect_color, self.detected_object_rect_thickness)
            
            
            label = '%s:' % (self._labels[self.__classIds[i]])
            labelSize, baseLine = cv2.getTextSize(
                label, self.label_font, self.label_fontscale, self.label_font_thickness)
            top = max(top, labelSize[1])
            cv2.rectangle(self.__frame, (left, top - round(1.5*labelSize[1])), (left + round(
                1.5*labelSize[0]), top + baseLine), self.label_rect_color, cv2.FILLED)
            cv2.putText(self.__frame, label, (left, top),
                        self.label_font, self.label_fontscale, self.label_text_color, self.label_font_thickness)
            
            self.__critical_dists = find_min_distance(self.__centers)
            for dist in self.__critical_dists:
                cv2.line(self.__frame, dist[0], dist[1],
                         self.critical_line_color, self.critical_line_thickness)
                mixer.init()
                #for producing alarm when critical distance
                mixer.music.load('kd.ogg')
                mixer.music.play()
        #text = "# 6ft violations: " + str(int(count))
        #cv2.putText(self.__frame, text)
