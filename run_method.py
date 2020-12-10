import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from scipy.spatial import distance
from draw import draw_circle, draw_rectangle
from detector import segmentation_imgSet2_imgSet3
from tracker import track_cells, drawLinesBetweenTwoImages
from analysis import cell_analysis,single_cell_analysis
from u_net import u_net, post_precessing

def read(data,sequence):
    imgs = []
    for root, dirs, files in os.walk("./data/" + data + sequence, topdown=False):
        files.sort()
        i=0
        for name in files:
            path = os.path.join(root, name)
            img = cv2.imread(path)
#           print(path)
            imgs.append(img)
            i+=1

    return imgs

# if dataset = 2 , the threshold of rectangle area = 20,  the threshold of radius = 6, detect_divide=0.15
# if dataset = 3, the threshold of rectangle area = 10, the threshold of radius = 2, detect_divide=0.5
def determine_parameter(data):
    if data == 'DIC-C2DH-HeLa/':
        radius = 2
        area = 10
        detect_divide =0.5
    if data == 'Fluo-N2DL-HeLa/':
        radius = 6
        area =20
        detect_divide = 0.15
    if data == 'PhC-C2DL-PSC/':
        radius = 1
        area = 10
        detect_divide = 0.5
    return radius, area, detect_divide

def Task1(data, sequence,radius,area, detect_divide):
    # print(data)
    if (data == 'Fluo-N2DL-HeLa/'):
        kp_list = []
        for i in range(len(sequence)-1):
            img1 = sequence[i]
            img2 = sequence[i+1]
            segmented_img1 = segmentation_imgSet2_imgSet3(img1, data)
            segmented_img2 = segmentation_imgSet2_imgSet3(img2, data)
            contours1 = cv2.findContours(segmented_img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours2 = cv2.findContours(segmented_img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            retangle_img1,count1  = draw_rectangle(segmented_img1, contours1[1], area, True)
            retangle_img2,count2 = draw_rectangle(segmented_img2, contours2[1], area,True)
            _, cells,_,last = draw_circle(segmented_img1, contours1[1], radius, detect_divide,True)
            cv2.drawContours(retangle_img2, last, -1, (0, 0, 255), 1)

            # get cell centers for both img1 and img2
            cell_centers = []
            for c in cells:
                center = c.get_center()
                cell_centers.append(center)

            kp1 = cv2.KeyPoint_convert(cell_centers)
            kp_list.append(kp1)
            curr_frame_with_kp = cv2.drawKeypoints(retangle_img1, kp1, retangle_img1, color=(255, 0, 0))
            cv2.imshow("current frame", curr_frame_with_kp)

            # find cells in cells 2
            tracked_points = track_cells(cell_centers, segmented_img1, segmented_img2)
            # draw tracked points in next image
            kp2 = cv2.KeyPoint_convert(tracked_points)

            next_frame_with_kp = cv2.drawKeypoints(retangle_img2, kp2, retangle_img2, color=(255, 0, 0))
            next_frame_with_kp = cv2.drawKeypoints(next_frame_with_kp, kp1, next_frame_with_kp, color=(0, 0, 255))
            point_color = (0, 0, 255)  # BGR
            thickness = 3
            lineType = 8
            for i in range(len(kp1)):
                cv2.line(next_frame_with_kp, (int(kp1[i].pt[0]), int(kp1[i].pt[1])),(int(kp2[i].pt[0]), int(kp2[i].pt[1])), point_color, thickness, lineType)

            kp_list.append(kp1)

            print("The number of cell detected  is " + str(count2))
            cv2.imshow("frame_with_matching", next_frame_with_kp)
            cv2.waitKey(0)
    else:
        print("U-net")
        tf.get_logger().setLevel('ERROR')
        tf.get_logger().warning('test')
        if (data == 'DIC-C2DH-HeLa/'):
            tf_model_path = "./model_1/"

        if (data == 'PhC-C2DL-PSC/'):
            tf_model_path = "./model_3/"

        loaded = tf.saved_model.load(tf_model_path)

        for i in range(len(sequence)-1):
            segmented_img1,original1 = u_net(sequence[i], data,loaded)
            segmented_img2, original2 = u_net(sequence[i+1], data,loaded)


            _, contours, _= cv2.findContours(segmented_img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            rectangle_img1, count1 = draw_rectangle(original1, contours, area,False)
            rectangle_img2, count2 = draw_rectangle(original1, contours, area, False)

            circle_img, cells, divide, last = draw_circle(segmented_img1, contours, radius, detect_divide, False)
            cv2.drawContours(rectangle_img2, last, -1, (0, 0, 255), 1)
            # get cell centers for both img1 and img2
            cell_centers = []
            for c in cells:
                center = c.get_center()
                cell_centers.append(center)

            kp1 = cv2.KeyPoint_convert(cell_centers)
            curr_frame_with_kp = cv2.drawKeypoints(rectangle_img1, kp1, rectangle_img1, color=(255, 0, 0))
            cv2.imshow("current frame",  curr_frame_with_kp)
            cv2.waitKey(0)

            tracked_points = track_cells(cell_centers, segmented_img1, segmented_img2)
            kp2 = cv2.KeyPoint_convert(tracked_points)

            next_frame_with_kp = cv2.drawKeypoints(rectangle_img2, kp2, rectangle_img2, color=(255, 0, 0))
            next_frame_with_kp = cv2.drawKeypoints(next_frame_with_kp, kp1, next_frame_with_kp, color=(0, 0, 255))

            point_color = (0, 0, 255)  # BGR
            thickness = 3
            lineType = 8
            for i in range(len(kp1)):
                cv2.line(next_frame_with_kp, (int(kp1[i].pt[0]), int(kp1[i].pt[1])),
                         (int(kp2[i].pt[0]), int(kp2[i].pt[1])), point_color, thickness, lineType)


            print("The number of cell detected  is " + str(count1))
            cv2.imshow("frame_with_matching", next_frame_with_kp)
            cv2.waitKey(0)

def Task2(data,sequence,radius,area, detect_divide):
    #print(data)
    i=0
    if (data == 'Fluo-N2DL-HeLa/'):
        for img in sequence:
            #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            segmented_img = segmentation_imgSet2_imgSet3(img, data)
            # cv2.imshow("segmented img",segmented_img)
            # cv2.waitKey(0)
            contours = cv2.findContours(segmented_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            circles, cells, count,_ = draw_circle(segmented_img, contours[1], radius, detect_divide,True)
            print("The number of dividing cell detected in current frame is " + str(count))
            cv2.imshow("img", circles)
            cv2.waitKey(0)
            i += 1

    else:
        tf.get_logger().setLevel('ERROR')
        tf.get_logger().warning('test')
        if (data == 'DIC-C2DH-HeLa/'):
            tf_model_path = "./model_1/"

        if (data == 'PhC-C2DL-PSC/'):
            tf_model_path = "./model_3/"

        i=0
        loaded = tf.saved_model.load(tf_model_path)
        for img in sequence:
            segmented_img,original = u_net(img, data,loaded)
            _, contours,_ = cv2.findContours(segmented_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            circle_img, cells, divide, last = draw_circle(original, contours, radius, detect_divide,False)
            print("The number of dividing cell detected in current frame is " + str(divide))
            i++1
            cv2.imshow("current frame", circle_img)
            cv2.waitKey(0)


# start here
data1 = 'DIC-C2DH-HeLa/'
data2 = 'Fluo-N2DL-HeLa/'
data3 = 'PhC-C2DL-PSC/'
s1 = 'Sequence 1'
s2 = 'Sequence 2'
s3 = 'Sequence 3'
s4 = 'Sequence 4'

# different thresholds for different dataset
#radius, area, detect_divide = determine_parameter(data1)
radius, area, detect_divide = determine_parameter(data2)
#radius, area, detect_divide = determine_parameter(data3)

# select one of the datasets and choose a sequence
#sequence4 = read(data1,s4)
sequence3 = read(data2,s3)
#sequence3 = read(data3,s3)

# run each task

#Task1(data1,sequence4,radius,area, detect_divide)
Task1(data2,sequence3,radius,area, detect_divide)
#Task1(data3,sequence3,radius,area, detect_divide) #color

#Task2(data1,sequence4,radius,area, detect_divide)
#Task2(data2,sequence3,radius,area, detect_divide)
#Task2(data3,sequence3,radius,area, detect_divide)


# click one cell on the image first, then program would keep tacking and print information of selected cell
def Task3(start, end, data,sequence):
    single_cell_analysis(start, end, data,sequence)

#Task3(49,90,data2,s3)
cv2.waitKey(0)
cv2.destroyAllWindows()