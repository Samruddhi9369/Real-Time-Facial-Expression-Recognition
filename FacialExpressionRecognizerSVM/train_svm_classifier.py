import pandas
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import numpy as np

data = pandas.read_csv('dataset/fer2013.csv') # read and store dataset in numpy arrays
labels = np.array(data['emotion'])
pixels = np.array(data['pixels'])
image_data = []
for i in range(len(pixels)):
    img_data = pixels[i].split()
    img_data = np.array(img_data).astype(int)
    image_data.append(img_data)
image_data = np.array(image_data)

# Split image dataset into 80% training dataset and 20% classification dataset
def split_datasets():
    training_set_size = 0.8
    test_set_size = 1-training_set_size
    rng_state = np.random.get_state()
    np.random.shuffle(labels)
    np.random.set_state(rng_state)
    np.random.shuffle(image_data)
    training_data = image_data[:int(len(image_data)*training_set_size)]
    training_labels = labels[:int(len(labels)*training_set_size)]
    prediction_data = image_data[-int(len(image_data)*test_set_size):]
    prediction_labels = labels[-int(len(labels)*test_set_size):]
    return training_data, training_labels, prediction_data, prediction_labels

# Training of FisherFace classifier over training and prediction dataset which returns correctness percentage after predicition of images
def train_classifier():
    training_data, training_labels, prediction_data, prediction_labels = split_datasets()

    print("Training SVM Classifier")
    print("Size of training dataset is:", len(training_labels), "images")
    linearSVC = LinearSVC() # creating linear svm model

    linearSVC.fit(np.asarray(training_data), np.asarray(training_labels)) #train model
    joblib.dump(linearSVC, 'linear_face_svm.pkl') # pickling the trained model in linear_face_svm.pkl file
    print("Predicting Classification Set")
    predictions = linearSVC.predict(np.asarray(prediction_data)) #predict labels of prediction dataset
    correct = predictions - np.asarray(prediction_labels)
    correct = correct[correct == 0]
    print("Accuracy Score: %f" % accuracy_score(predictions,np.asarray(prediction_labels)))
    return len(correct)/len(prediction_labels) # Return accuracy score of prediction

# Traning of SVM model
percent = train_classifier()
print("Percent correct: %f" % percent)
