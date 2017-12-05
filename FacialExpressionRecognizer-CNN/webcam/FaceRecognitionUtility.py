import cv2
import numpy as np

CASCADE_PATH = "haarcascade_frontalface_default.xml"

RESIZE_SCALE = 3

# returns face coordinates from the image
def fnGetFaceCoordinates(image):
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    rectangles = cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(48, 48)
        )

    # For now, we only deal with the case that we detect one face.
    if(len(rectangles) != 1) :
        return None
    
    face = rectangles[0]
    bounding_box = [face[0], face[1], face[0] + face[2], face[1] + face[3]]

    return bounding_box

# Draws rectangle around the face in passed image using passed faceCoordinates.
def fnDrawFace(image, faceCoordinates, Rectangle_Color = (0, 255, 0)):
    cv2.rectangle(np.asarray(image), (faceCoordinates[0], faceCoordinates[1]), \
                  (faceCoordinates[2], faceCoordinates[3]), Rectangle_Color, thickness=2)

# Crops the image to return only face
def fnCropFace(image, faceCoordinates):
    return image[faceCoordinates[1]:faceCoordinates[3], faceCoordinates[0]:faceCoordinates[2]]

# This function will crop user's face from the original frame
def fnPreprocessImage(image, faceCoordinates, face_shape=(48, 48)):
    face = fnCropFace(image, faceCoordinates)
    face_scaled = cv2.resize(face, face_shape)
    face_gray = cv2.cvtColor(face_scaled, cv2.COLOR_BGR2GRAY)
    return face_gray
