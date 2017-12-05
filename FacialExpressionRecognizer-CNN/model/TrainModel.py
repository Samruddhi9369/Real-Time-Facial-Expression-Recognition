import argparse

import cv2
import numpy as np

from keras.callbacks import LambdaCallback, EarlyStopping

import FeatureUtility as fu
import CNN_model

parser = argparse.ArgumentParser(description=("Model training process."))
parser.add_argument('--test', help=("Input a single image to check if the model works well."))

args = parser.parse_args()

def main():
    # Initialize Model
    model = CNN_model.fnBuildModel()

    #If user passes a single image to check if the model works
    if args.test is not None:
        print ("---------- In Testing mode")
        image = cv2.imread(args.test)
        image = fu.preprocessing(image)
        image = np.expand_dims(image, axis=0)
        y = np.expand_dims(np.asarray([0]), axis=0)
        BatchSize = 1
        model.fit(image, y, epochs=400, \
                batch_size=BatchSize, \
                validation_split=0.1, \
                shuffle=True, verbose=0)
        return


    X_filename = '../data/X_train_train.npy'
    Y_filename = '../data/y_train_train.npy'
    X_train = np.load(X_filename)
    Y_train = np.load(Y_filename)
    print(X_train.shape)
    print(Y_train.shape)
   
    print("Training started...........")

    arrCallbacks = []
    earlystop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    batch_print_callback = LambdaCallback(on_batch_begin=lambda batch, logs: print(batch))
    epoch_print_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print("epoch:", epoch))
    arrCallbacks.append(earlystop_callback)
    arrCallbacks.append(batch_print_callback)
    arrCallbacks.append(epoch_print_callback)

    BatchSize = 512
    hist =  model.fit(X_train, Y_train, epochs=500, \
            batch_size=BatchSize, \
            validation_split=0.3, \
            shuffle=True, verbose=0, \
            callbacks=arrCallbacks)

    model.save_weights('my_model_weights.h5')
    #scores = model.evaluate(X_train, Y_train, verbose=0)
    # model result:
    train_val_accuracy = hist.history
    # Get and print training accuracy
    train_accuracy = train_val_accuracy['acc']
    # Get and print validation accuracy
    val_accuracy = train_val_accuracy['val_acc']
    print ("Done!")
    print ("Train acc: %.3f" % train_accuracy[-1])
    print ("Validation acc: %.3f" % val_accuracy[-1])

    #print ("Train loss : %.3f" % scores[0])
    #print ("Train accuracy : %.3f" % scores[1])
    print ("Training finished")

if __name__ == "__main__":
    main()
