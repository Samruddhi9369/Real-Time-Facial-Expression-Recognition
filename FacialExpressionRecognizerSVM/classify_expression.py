import cv2
import numpy as np
from sklearn.externals import joblib

#Facial Expression Classifier For SVM model
class FacialExpressionClassifier:
    def __init__(self):
        self.expressions = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  #List of facial expressions
        self.model = joblib.load('linear_face_svm.pkl') #Unpickling the model saved in linear_face_svm.pkl file
        
    def classify_expression(self, image):
        image = np.reshape(image, (1, 2304)) #2304 is the sze of the input image with shape (48x48)
        return int(self.model.predict(image)) # returns facial expression predicted by SVM model
