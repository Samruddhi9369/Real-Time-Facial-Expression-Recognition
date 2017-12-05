# We chose Convolutional neural network to build my model architecture.
# This architecture contain an input layer, some convolutional layers, some dense layers and output layer.
import os, sys
module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path)

# Used Keras libraries to create model
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization


from keras import backend as K
K.set_image_dim_ordering('th')

# This model contains 9 layer deep in convolution with one max-pooling after every 3 convolutional layers
# followed by 2 dense layer and finally one output layer.
def fnBuildModel(preCalculatedWeightPath=None, shape=(48, 48)):
    # In Keras, model is created as Sequential() and more layers are added to build architecture.
    model = Sequential()

    model.add (ZeroPadding2D ((1, 1), input_shape=(1, 48, 48)))
    model.add (Conv2D (32, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1), input_shape=(1, 48, 48)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add (ZeroPadding2D ((1, 1)))
    model.add (Conv2D (64, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    # 20% dropout
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    
    print ("Create model successfully")
    if preCalculatedWeightPath:
        model.load_weights(preCalculatedWeightPath)

    model.compile(optimizer='adam', loss='categorical_crossentropy', \
        metrics=['accuracy'])

    return model
