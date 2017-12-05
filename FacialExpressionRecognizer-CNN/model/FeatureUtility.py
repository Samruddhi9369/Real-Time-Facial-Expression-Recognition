import sys, os
import cv2
import numpy as np


# This function resize images to 48-by-48-pixel grayscale images
def fnPreprocessImage(image, size=(48, 48)):
    image = cv2.cvtColor (image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize (image, size).astype (np.float32)
    return image


# This function extracts features from images available at [strPath]
def fnExtractFeatures(strPath):
    x, y = [], []
    label = 0
    for dir in os.listdir (strPath):
        print (dir)
        strSubPath = os.path.join (strPath, dir)
        print (strSubPath)
        for file in os.listdir (strSubPath):
            print (file)
            strFilePath = os.path.join (strSubPath, file)
            image = cv2.imread (strFilePath)
            image = fnPreprocessImage (image)
            x.append (image)

            emoClassLabels = [0, 0, 0, 0, 0, 0, 0]
            emoClassLabels[label] = 1
            y.append (emoClassLabels)
        label += 1

    x = np.asarray (x)
    y = np.asarray (y)
    return x, y


if __name__ == "__main__":
    X, Y = fnExtractFeatures (sys.argv[1])
    print (X, Y)
    print (type (X), type (X[0]))
    print (X[212])
    print (len (X))
