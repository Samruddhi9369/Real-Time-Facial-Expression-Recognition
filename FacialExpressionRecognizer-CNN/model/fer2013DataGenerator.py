from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
import random
import sys

# This file separate training and validation data. While generating data, we classified Disgust as Angry.
# So resulting data will contains 6-class balanced dataset that contains Angry, Fear, Happy, Sad, Surprise and Neutral
# fer2013 dataset:
# It comprises a total of 35887 pre-cropped, 48-by-48-pixel grayscale images of faces each
# labeled with one of the 7 emotion classes: anger, disgust, fear, happiness, sadness, surprise, and neutral.
# Training       28709
# PrivateTest     3589
# PublicTest      3589

# emotion labels from FER2013:
original_emo_classes = {'Angry': 0,
                        'Disgust': 1,
                        'Fear': 2,
                        'Happy': 3,
                        'Sad': 4,
                        'Surprise': 5,
                        'Neutral': 6}
final_emo_clasees = ['Angry',
                     'Fear',
                     'Happy',
                     'Sad',
                     'Surprise',
                     'Neutral']

# Reconstruct original image to size 48X48. Returns numpy array of image pixels
def fnReconstruct(original_pixels, size=(48, 48)):
    arrPixels = []
    for pixel in original_pixels.split():
        arrPixels.append(int(pixel))
    arrPixels = np.asarray(arrPixels)
    return arrPixels.reshape(size)

#This function merge disgust emotion label to anger label and returns count of each emotion class
def fnGetEmotionCount(y_train, emoClasses, verbose=True):
    emo_classcount = {}
    #fer2013 dataset contains only 113 samples of "disgust" class compared to many other classes.
    #Therefore we merge disgust into anger to prevent this imbalance.
    print ('Disgust classified as Angry')
    y_train.loc[y_train == 1] = 0
    emoClasses.remove('Disgust')
    for newNum, className in enumerate(emoClasses):
        y_train.loc[(y_train == original_emo_classes[className])] = newNum
        class_count = sum(y_train == (newNum))
        if verbose:
            print ('{}: {} with {} samples'.format(newNum, className, class_count))
        emo_classcount[className] = (newNum, class_count)
    return y_train.values, emo_classcount

#loads data from fer2013.csv
def fnLoadData(Sample_split_fraction=0.3, usage='Training', boolCategorize=True, verbose=True,
               default_classes=['Angry', 'Happy'], filepath='../data/fer2013.csv'):
    # read .csv file using pandas library
    df = pd.read_csv(filepath)
    df = df[df.Usage == usage]
    arrFrames = []
    default_classes.append('Disgust')
    for _class in default_classes:
        class_df = df[df['emotion'] == original_emo_classes[_class]]
        arrFrames.append(class_df)
    data = pd.concat(arrFrames, axis=0)
    rows = random.sample(list(data.index), int(len(data) * Sample_split_fraction))
    data = data.ix[rows]
    print ('{} set for {}: {}'.format(usage, default_classes, data.shape))
    data['pixels'] = data.pixels.apply(lambda x: fnReconstruct(x))
    x = np.array([mat for mat in data.pixels])
    X_train = x.reshape(-1, 1, x.shape[1], x.shape[2])
    Y_train, new_dict = fnGetEmotionCount(data.emotion, default_classes, verbose)
    print (new_dict)
    if boolCategorize:
        Y_train = to_categorical(Y_train)
    return X_train, Y_train, new_dict

# Save X_train (images) and Y_train (labels) to local folder for training
def fnSaveData(X_train, Y_train, fname='', folder='../data/'):
    np.save(folder + 'X_train' + fname, X_train)
    np.save(folder + 'Y_train' + fname, Y_train)

if __name__ == '__main__':
    # makes the numpy arrays ready to use:
    print ('Making moves...')
    final_emo_clasees = ['Angry',
                         'Fear',
                         'Happy',
                         'Sad',
                         'Surprise',
                         'Neutral']

    X_train, Y_train, emo_dict = fnLoadData(Sample_split_fraction=1.0,
                                            default_classes=final_emo_clasees,
                                            usage='Training',
                                            verbose=True)
    print ('Saving...')
    fnSaveData(X_train, Y_train, fname='_train')
    print (X_train.shape)
    print (Y_train.shape)
    print ('Done!')
