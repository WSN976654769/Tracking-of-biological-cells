import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from cell import Cell


def draw_circle(img, contours, r,v,flag):
   # print(img.shape)
    # if dataset2, true
    if flag == True:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    list = []
    id = 0
    count = 0
    last =[]
    for cnt in contours:
        # get the min enclosing circle
        [x, y], radius = cv2.minEnclosingCircle(cnt)
        # convert all values to int
        center = [int(x), int(y)]
        radius = int(radius)
        #B
        if radius > r:

            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True) #small
            hull = cv2.convexHull(cnt) #big
            #cv2.drawContours(img, [cnt], -1, (255, 0, 0), 2)    #B
            last.append(cnt)
            ret = cv2.matchShapes(hull, approx, 1, 0.0)
            # print(ret)
            # detect devision
            if ret< v:
                cv2.drawContours(img, [approx], -1, (0, 255, 0), 2) #G
            else:
                cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)  # R
                count+=1

            # record information of each cell, append it to list
            cell = Cell(id, center, radius)
            list.append(cell)
            id += 1

     # cv2.imshow("img_with_circle", img)
    # cv2.waitKey(0)
    # print("The number of detected cell is:")
    # print(id + 1)

    return img, list,count,last


def draw_rectangle(img, contours, a,flag):
    #print(img.shape)
    if flag == True:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    count = 0
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > a:
            # print(area)
            # draw a green rectangle to visualize the bounding rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # get the min area rect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            count += 1

    # cv2.imshow("img_with_rectangle", img)
    # cv2.waitKey(0)
    # print("The number of detected cell is: ")
    # print(id + 1)
    return img,count


