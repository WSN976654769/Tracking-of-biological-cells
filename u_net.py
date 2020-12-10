import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
# noinspection PyUnresolvedReferences
from sklearn.cluster import MeanShift
from skimage.morphology import h_maxima
from draw import draw_circle, draw_rectangle

def u_net(img, data,loaded):
    tf.get_logger().setLevel('ERROR')
    tf.get_logger().warning('test')

    #imgset_no = 1
    if ( data == 'DIC-C2DH-HeLa/'):
        #imgset = './data/DIC-C2DH-HeLa'
        tf_model_path = "./model_1/"
        w, h = 256, 256


    if ( data == 'PhC-C2DL-PSC/'):
        #imgset = './data/PhC-C2DL-PSC'
        tf_model_path = "./model_3/"
        w, h = 720, 576

    # model
    #loaded = tf.saved_model.load(tf_model_path)
    original =  cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img.shape)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # prepare_and_predict
    img = cv2.resize(img, (w, h))
    X = np.reshape(img, (1, h, w, 1))
    #print(X.shape)
    X = (X / 255).astype(np.float32)

    npy_tensor = loaded.signatures['serving_default'](tf.convert_to_tensor(X))['output'].numpy()
    npy = np.reshape(npy_tensor, (h, w)) * 255
    npy = np.array(npy, dtype="u1")

    # output image
    #cv2.imshow("img",img)
    #cv2.waitKey(0)

    final_img = post_precessing(data, npy,img)
    # cv2.imshow("img", circles)
    return final_img,original

######################################################
def post_precessing(data,npy,img):
    if (data == 'DIC-C2DH-HeLa/'):
        #print("1: GaussianBlur -- H-maxima -- Opening -- Find contours")
        img_blur = cv2.GaussianBlur(npy, (15, 15), 0)
        # cv2.imshow("img_blur", img_blur)
        # cv2.waitKey(0)

        img_h = h_maxima(img_blur, 20, selem=None)
        img_h = img_h * 255
        # print(img_h.shape)
        # cv2.imshow("img_h", img_h)
        # cv2.waitKey(0)

        kernel = np.ones((7, 7), np.uint8)
        img_h_opening = cv2.dilate(img_h, kernel)
        # cv2.imshow("img_open", img_h_opening)
        # cv2.waitKey(0)
        img = img_h_opening

    if (data == 'PhC-C2DL-PSC/'):
        #print("3: Threshold -- Opening -- Watershed")
        thresh = cv2.threshold(npy, 127, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow("thresh", thresh)
        # cv2.waitKey(0)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # cv2.imshow("opening", opening)
        # cv2.waitKey(0)

        distance = ndi.distance_transform_edt(opening)
        local_max = peak_local_max(distance, indices=False, min_distance=10, labels=opening)
        markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]
        ws_labels = watershed(-distance, markers=markers, mask=opening)
        ws_labels = ws_labels *255
        ret, ws_labels= cv2.threshold(ws_labels.astype(np.uint8), 5, 255, cv2.THRESH_BINARY)
        img = ws_labels
        # cv2.imshow("final",  ws_labels)
        # cv2.waitKey(0)


    return img