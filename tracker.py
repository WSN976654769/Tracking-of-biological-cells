import numpy as np
import cv2


 # Track a list points in next frame using optical flow, return a list of corresponding points in next frame
# cell_centers: a list of cell center in time frame t1  [[x1,y2], [x2,y2] ... [xi, yi]]
# old_img_frame: image in time frame t1, gray scale img
# new_img_frame :  image in time frame t2, gray scale img
def track_cells(cell_centers, old_img_frame, new_img_frame):
    old_points = np.array(cell_centers, dtype=np.float32)
    lk_params = dict(winSize = (20, 20),
    maxLevel = 4,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_img_frame, new_img_frame, old_points, None, **lk_params)
    # new_points = np.uint32(new_points)
    new_points = new_points.tolist()
    return new_points


# len of kp1 = len kp2
# kp1[i] connect to to kp2[i]

# curr_frame: image1
# kp1: key points to matches
# next_frame : image2
# kp2: key points to be matched to
def drawLinesBetweenTwoImages(curr_frame, kp1, next_frame, kp2):
    matches = [cv2.DMatch() for i in range(len(kp1))]
    j = 0
    for i in matches:
        i.queryIdx = j
        i.trainIdx = j
        j += 1

    result = cv2.drawMatches(curr_frame, kp1, next_frame, kp2, matches, None)
    return result



