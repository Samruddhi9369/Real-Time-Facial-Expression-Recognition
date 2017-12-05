import os
import cv2
import pandas
import random
import numpy as np

fisherface = cv2.face.createFisherFaceRecognizer()  # Initialize OpenCV's fisher face classifier
emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # Emotion list

data = pandas.read_csv('dataset/fer2013.csv')
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
def train_classifier(num_run=0):
    training_data, training_labels, prediction_data, prediction_labels = split_datasets()

    print("Training of Fisher Face Classifier")
    print("Size of training dataset is:", len(training_labels), "images")
    fisherface.train(np.asarray(training_data), np.asarray(training_labels))
    fisherface.save('fish_models/fish_model' + str(num_run) + '.xml') # saving all fisherface model after training the dataset
    print("Predicting Classification Set")
    count = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred = fisherface.predict(image)
        if pred == prediction_labels[count]:
            correct += 1
            count += 1
        else:
            incorrect += 1
            count += 1
    return (100 * correct) / (correct + incorrect)


# Run recognizer on training and prediction dataset
metascore = []
for i in range(0, 10):
    correct = train_classifier(num_run=i)
    print("got", correct, "percent correct!")
    metascore.append(correct)

# Taking the mean of correctness of 10 Fisherface models
print("Average score:", np.mean(metascore), "percent correct!")