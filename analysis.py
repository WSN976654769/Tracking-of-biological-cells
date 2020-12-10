import cv2
import numpy as np
from detector import segmentation_imgSet2_imgSet3
from tracker import track_cells, drawLinesBetweenTwoImages
from cell import find_cells, find_cells_id, find_cells_centers, Cells_map,find_cell_by_position
import copy
from cell_select import Mouse_click

def get_img_pathes(start_frame, end_frame, image_set, sequence_num):
    img_frames = []
    for i in range(start_frame, end_frame+1):
        path = './data/' + image_set  + str(sequence_num) + "/"
        img = ""
        if i < 10:
            img = "t00" + str(i) + ".tif"
        if 10 <= i < 99:
            img = "t0" + str(i) + ".tif"
        if i >= 100:
            img = "t" + str(i) + ".tif"
        path = path + img
        img_frames.append(path)
    return img_frames


# start frame : int
# end frame : int
# image set: data1 or data2 or data 3
# seq: s1, s2 , s3 or s4

def cell_analysis(start_frame, end_frame, image_set, sequence_num):

    pathes = get_img_pathes(start_frame,end_frame,image_set,sequence_num)
    curr_img = cv2.imread(pathes[0], 1)

    # img1 = cv2.imread(pathes[0], 1)
    # cv2.imshow("o1", img1)
    # img1 = segmentation_imgSet2_imgSet3(img1, image_set)
    #
    # img2 = cv2.imread(pathes[-1], 1)
    # cv2.imshow("o2", img2)
    # img2 = segmentation_imgSet2_imgSet3(img2, image_set)
    #
    # kp1 = cv2.KeyPoint_convert([[199,611]])
    # p1 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 0))
    # kp2 = cv2.KeyPoint_convert([[188, 606]])
    # p2 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 0))

    # cv2.imshow("p1",p1)
    # cv2.imshow("p2", p2)

    curr_seg_img = segmentation_imgSet2_imgSet3(curr_img, image_set)
    curr_cells = find_cells(curr_seg_img, start_frame)

    # create cell map for first frame
    curr_ids = find_cells_id(curr_cells)
    curr_centers = find_cells_centers(curr_cells)

    # kp1 = cv2.KeyPoint_convert(curr_centers)
    # p1 = cv2.drawKeypoints(curr_seg_img, kp1, curr_seg_img, color=(255, 0, 0))
    # cv2.imshow("p1", p1)


    # all cell has 0 distance at the beginning
    curr_net_distance = curr_total_distance = [0 for i in range(len(curr_ids))]
    cells_map = Cells_map(curr_ids,curr_centers,curr_net_distance,curr_total_distance,start_frame)


    #tracked cells in first frame
    tracked_cells = cells_map
    initial_cells = cells_map
    previous_cells = initial_cells

    for i in range(len(pathes)-1):

        curr_centers = tracked_cells.get_cells_centers()
        curr_ids = tracked_cells.get_cells_ids()
        cell_num = len(curr_ids)

        # find segmented img of next frame
        next_img = cv2.imread(pathes[i+1], 1)
        next_seg_img = segmentation_imgSet2_imgSet3(next_img, image_set)

        # find centers in next frame by optical flow
        tracked_centers= track_cells(curr_centers, curr_seg_img, next_seg_img)

        next_ids = curr_ids

        # kp = cv2.KeyPoint_convert(tracked_centers)
        # p = cv2.drawKeypoints(next_seg_img, kp, next_seg_img, color=(255, 0, 0))
        # cv2.imshow("p" + str(i + 1), p)

        # find total distance
        new_travel_dist = []
        prev_total_dist = previous_cells.get_total_ditance()
        prev_cell_centers = previous_cells.get_cells_centers()
        for i in range(cell_num):
            dist = np.linalg.norm(np.array(prev_cell_centers[i]) - np.array(tracked_centers[i]))
            new_travel_dist.append(dist)
        new_total_dist = [round(a + b, 3) for a, b in zip (prev_total_dist, new_travel_dist)]

        # calculate net distance
        new_net_dist = []
        centers_start_frames = initial_cells.get_cells_centers()
        for i in range(cell_num):
            dist = np.linalg.norm(np.array(centers_start_frames[i]) - np.array(tracked_centers[i]))
            new_net_dist.append(round(dist,3))



        tracked_cells = Cells_map(next_ids,tracked_centers,new_net_dist,new_total_dist, start_frame+i+1)
        previous_cells = copy.deepcopy(tracked_cells)
        curr_seg_img = next_seg_img



    cell_ids = tracked_cells.get_cells_ids()
    centers_start_frames = initial_cells.get_cells_centers()
    centers_last_frames = tracked_cells.get_cells_centers()
    net_dist = tracked_cells.get_net_distance()
    total_dist = tracked_cells.get_total_ditance()
    C_ratio = []
    speed = []

    for i in range(len(cell_ids)):
        C_ratio.append(round(total_dist[i]/net_dist[i],3))
        speed.append(round(net_dist[i]/(end_frame-start_frame),3))

    for i in range(len(cell_ids)):
        last_pos = centers_last_frames[i]
        last_pos = [round(last_pos[0]), round(last_pos[1])]
        print("cell Id :{0}, start pos {1} in frame {2}, "
              "end pos {3} in frame {4}, net dist: "
              "{5}, total dist {6}, travel speed: {7}, Confinement ration: {8}".format(cell_ids[i],
               centers_start_frames[i],start_frame, last_pos,
                                                end_frame, net_dist[i],total_dist[i], speed[i], C_ratio[i]))


    # img2 = cv2.imread(pathes[-1], 1)
    # img2 = segmentation_imgSet2_imgSet3(img2, image_set)
    # kp2 = cv2.KeyPoint_convert(centers_last_frames)
    # p2 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 0))
    # cv2.imshow("p2", p2)




def draw_center_and_bounding_box(img, tracked_center):
    contour, center = find_cell_by_position(img, tracked_center)
    x, y, w, h = cv2.boundingRect(contour)
    img_selecting = cv2.rectangle(img, (x, y), (x + w, y + h), (100, 109, 71), 2)
    kp = cv2.KeyPoint_convert([center])
    img = cv2.drawKeypoints(img_selecting, kp, img_selecting, color=(255, 0, 0))
    return img


def single_cell_analysis(start_frame, end_frame, image_set, sequence_num):
    pathes = get_img_pathes(start_frame, end_frame, image_set, sequence_num)
    curr_img = cv2.imread(pathes[0], 1)
    curr_seg_img = segmentation_imgSet2_imgSet3(curr_img, image_set)
    img_selecting = copy.deepcopy(curr_seg_img)
    m = Mouse_click()
    m.choose_cell(img_selecting)
    point = m.points[0]
    contour, center = find_cell_by_position(img_selecting, point)
    if center == None:
        print("please click on one cell")
        return
    print("You selected cell in postion {0}".format(center))
    x, y, w, h = cv2.boundingRect(contour)
    img_selecting = cv2.rectangle(img_selecting, (x, y), (x + w, y + h), (100, 109, 71), 2)
    kp = cv2.KeyPoint_convert([center])
    img_selecting = cv2.drawKeypoints(img_selecting, kp, img_selecting, color=(255, 0, 0))
    #cv2.imshow('Task 3 cell motion analysis -- frame {0}'.format(start_frame), img_selecting)
    cv2.imshow('Task 3 cell motion analysis', img_selecting)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cell = Cell_analysis(center, 0, 0, start_frame)

    print("Tracked cell at position {0} in frame {1}, net dist: {2}, "
          "total dist {3}, travel speed: {4},"
          " Confinement ration: {5}".format(
        [round(cell.center[0]),round(cell.center[1])], cell.frame,
        cell.net_distance,
        cell.total_distance,
        cell.speed, cell.confinement))

    tracked_cell = cell
    initial_cell = cell
    previous_cell = initial_cell

    for i in range(len(pathes) -1):
        curr_center = tracked_cell.center
        # find segmented img of next frame
        next_img = cv2.imread(pathes[i + 1], 1)
        next_seg_img = segmentation_imgSet2_imgSet3(next_img, image_set)

        # find center in next frame by optical flow
        tracked_center = track_cells([curr_center], curr_seg_img, next_seg_img)

        #find total distance
        prev_total_dist = previous_cell.total_distance
        prev_cell_center = previous_cell.center
        dist = np.linalg.norm(np.array(prev_cell_center) - np.array(tracked_center[0]))
        new_total_dist = round(prev_total_dist + dist, 3)

        # calculate net distance
        dist = np.linalg.norm(np.array(initial_cell.center) - np.array(tracked_center[0]))
        new_net_dist = (round(dist, 3))

        tracked_cell = Cell_analysis(tracked_center[0], new_net_dist, new_total_dist, start_frame + i + 1)
        tracked_cell.calculate_speed(i+1)
        tracked_cell.calculate_confinement()
        tracked_cell.printme()
        img_show = copy.deepcopy(next_seg_img)
        img_show = draw_center_and_bounding_box(img_show,tracked_center)

        previous_cell = copy.deepcopy(tracked_cell)
        curr_seg_img = next_seg_img

        #cv2.imshow('Task 3 cell motion analysis -- frame {0}'.format(start_frame + i + 1), img_show)
        cv2.imshow('Task 3 cell motion analysis', img_show)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()





class Cell_analysis:
    def __init__(self, center, net_distance, total_distance, frame):
        self.center = center
        self.frame = frame
        self.net_distance = net_distance
        self.total_distance = total_distance
        self.speed = None
        self.confinement = None

    def calculate_speed(self, frameNum):
        self.speed = round(self.net_distance / frameNum, 3)

    def calculate_confinement(self):
        self.confinement = round(self.total_distance/self.net_distance,3)


    def printme(self):
        print("Tracked cell at position {0} in frame {1}, net dist: {2}, "
              "total dist {3}, travel speed: {4},"
              " Confinement ratio: {5}".format(
            [round(self.center[0]), round(self.center[1])], self.frame,
            self.net_distance,
            self.total_distance,
            self.speed, self.confinement))
