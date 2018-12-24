import cv2
import imutils


def preprocess(image, width, height):
    (h, w) = image.shape[ : 2]

    if h>w:
        imutils.resize(image, height=h)
    else:
        imutils.resize(image, width=w)

    padW = int((width - image.shape[1])/2)
    padH = int((height-image.shape[0])/2)

    #image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image,( width, height))

    return image