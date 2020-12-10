import cv2
import numpy as np


class Cells_map:
    def __init__(self, ids, centers, net_distance, total_distance, frame):
        self.cells_ids = ids
        self.cells_centers = centers
        self.frame = frame
        self.net_distance = net_distance
        self.total_distance = total_distance

    def get_cells_ids(self):
        return self.cells_ids

    def get_cells_centers(self):
        return self.cells_centers

    def get_frame(self):
        return self.frame

    def get_total_ditance(self):
        return self.total_distance

    def get_net_distance(self):
        return self.net_distance



class Cell:
    def __init__(self, id, center, frame_Num, radius="9999999"):
        self.id = int(id)
        self.radius = int(radius)
        self.center = center # [x, y]
        self.area = float(self.radius ** 2 * 3.14)
        self.frame = frame_Num

    def get_id(self):
        return self.id

    def get_radius(self):
        return self.radius

    # location
    def get_center(self):
        return self.center

    def get_area(self):
        return self.area



# given list of Cell class
# return cell centers as [[x,y], [x,y], [x,y] ...]
def find_cells_centers(cells):
    cell_centers = []
    for c in cells:
        center = c.get_center()
        cell_centers.append(center)
    return cell_centers


# given list of Cell class
# return cell centers as [id1, id2, id3 ...]
def find_cells_id(cells):
    ids = []
    for c in cells:
        id = c.get_id()
        ids.append(id)
    return ids


# given a img after segmentation, and frame number
# return a list of cell classes
def find_cells(segmented_img, frameNum):
    contours = cv2.findContours(segmented_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    id = 0
    for cnt in contours[1]:
        [x, y], radius = cv2.minEnclosingCircle(cnt)
        center = [int(x), int(y)]
        radius = int(radius)

        if radius > 1.2:
            cell = Cell(id, center, frameNum, radius)
            cells.append(cell)
            id += 1
    return cells



def find_cell_by_position(segmented_img, cell_postion):
    contours= cv2.findContours(segmented_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1]
    closest_cnt = contours[0]
    [x, y], radius = cv2.minEnclosingCircle(contours[0])
    closest_distance = np.linalg.norm(np.array([x,y]) - np.array(cell_postion))
    closest_center = [x,y]

    for cnt in contours[1:]:
        [x, y], radius = cv2.minEnclosingCircle(cnt)
        center = [x, y]
        distance = np.linalg.norm(np.array(center) - np.array(cell_postion))
        if distance < closest_distance:
            closest_distance = distance
            closest_cnt = cnt
            closest_center = center

    return closest_cnt,closest_center





