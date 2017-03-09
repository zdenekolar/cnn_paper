import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from matplotlib import pyplot as plt
from keras.regularizers import l1, l2
import keras
from vgg16_lower_part import vgg16_lower_part

train_data_dir = 'data/random1000'
validation_data_dir = 'data/validation_A'

weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
img_width, img_height = 300, 300

nb_train_samples = 4096
nb_validation_samples = 2944
nb_epoch = 50
dropout = 0.5
l2_c = 0.2


def save_bottleneck_features():
    '''
    Load trained VGG-16 network.
    Load weights.
    Extract the features from the last layer into bottleneck_features_train.npy
    Extarct the features from the last layer to bottleneck_features_validation.npy
    :return: None
    '''
    datagen = ImageDataGenerator(rescale=1./255)

    model = vgg16_lower_part(img_width, img_height, weights_path)

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)
    
    print('Bottleneck features saved.')

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)
    
    print('Bottleneck validation features saved.')


def train_top_model():
    '''
    Load the features from save_bottleneck_features.
    Create a new fully connected network (just like from the end of VGG-16).
    Train the network.
    :return:
    '''
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))
    
    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))    

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, W_regularizer= l2(l2_c), activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
    
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    cb = keras.callbacks.CSVLogger('log_epochs.csv', separator=',', append=False)
    history = model.fit(train_data, train_labels, verbose=2, callbacks=[cb],
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    return history


def display_stats():
    '''
    Plot the training statistics
    :return:
    '''
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    
if __name__ == '__main__':
    '''
    Entry point.
    '''
    save_bottleneck_features()
    history = train_top_model()
    display_stats()
