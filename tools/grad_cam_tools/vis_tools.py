import cv2
import numpy as np
def drawBboxes(Boxes, Image):
    for box in Boxes:
        curBox = [np.int(i) for i in box[:4]]
        score = np.float(box[4])
        cv2.rectangle(Image, (curBox[0], curBox[1]), (curBox[2], curBox[3]), (0, 255, 255), 2)
        cv2.putText(Image, "{:.3f}".format(score), (box[0], box[1]+20), 0, 0.8,  (0, 0, 255))