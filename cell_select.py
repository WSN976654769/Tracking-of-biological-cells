import cv2
import numpy as np


class Mouse_click:
    def __init__(self):
        self.points = []
        self.img = None


    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x,y])
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # strXY = str(x)+", "+str(y)
            # cv2.putText(self.img, strXY, (x,y), font, 0.5, (255,255,0), 2)
            # cv2.imshow("image", self.img)


    def choose_cell(self, img):
        self.img = img
        cv2.imshow("selecting cell", img)
        cv2.setMouseCallback("selecting cell", self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# data1 = 'DIC-C2DH-HeLa/'
# data2 = 'Fluo-N2DL-HeLa/'
# data3 = 'PhC-C2DL-PSC/'
# s1 = 'Sequence 1'
# s2 = 'Sequence 2'
# s3 = 'Sequence 3'
# s4 = 'Sequence 4'
#
# path = "./data/" + data2 + s1 +"/t010.tif"
# img = cv2.imread(path,1)
# cv2.imshow("1",img)
# cv2.waitKey(0)
# from tracker import track_cells
# result = track_cells([[100.32,120.23]],img, img)
# print(result)
