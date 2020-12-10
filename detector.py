import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from cell import Cell

# noise removal
def open_demo(image):
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel=kernel)
    return dst


def max_min(img):
    size = (9, 9)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # Applies the minimum filter with kernel NxN
    min_img = cv2.erode(img, kernel)

    # Applies the maximum filter with kernel NxN
    max_img = cv2.dilate(min_img, kernel)

    result = img.astype(np.int32) - max_img
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result


def segmentation_imgSet2_imgSet3(img, dataset):
    # cv2.imshow("orignal", img)
    # cv2.waitKey(0)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = open_demo(img)
    # max_min filtering preprocessing only used for image dataset 3
    # print(dataset)
    if dataset == "PhC-C2DL-PSC/":
        # print("dataset 3")
        img = max_min(img)

    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0)
    # Wastershed
    # Compute Euclidean distance from every binary pixel
    # to the nearest zero pixel then find peaks
    distance_map = ndi.distance_transform_edt(thresh)
    local_max = peak_local_max(distance_map, indices=False, labels=thresh)

    # Perform connected component analysis then apply Watershed
    markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]
    labels = watershed(-distance_map, markers, mask=thresh)

    ret, img_bin = cv2.threshold(labels.astype(np.uint8), 5, 255, cv2.THRESH_BINARY)
    # cv2.imshow("img", img_bin )
    # cv2.waitKey(0)

    return img_bin




