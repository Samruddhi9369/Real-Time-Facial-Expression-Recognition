import cv2
import numpy as np
from sklearn.externals import joblib
from scipy import stats

#Facial Expression Classifier For FisherFace model
class FacialExpressionClassifier:
    
    def __init__(self):
        self.expressions = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # Emotion list
        
        self.models = []
        for i in range(10):
            model_temp = cv2.face.createFisherFaceRecognizer()
            model_temp.load('fish_models/fish_model' + str(i) + '.xml') # Load all saved fisher face models
            self.models.append(model_temp)

    def classify_expression(self, image):
		emotion_guesses = np.zeros((len(self.models), 1))
		for index in range(len(self.models)):
			prediction = self.models[index].predict(image) # predict input image using each fisher face model
			emotion_guesses[index][0] = prediction
				
		return int(stats.mode(emotion_guesses)[0][0]) # returns facial expression predicted by all fisher face models