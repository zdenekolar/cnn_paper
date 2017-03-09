import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from matplotlib import pyplot as plt
from keras.regularizers import l1, l2
import keras
from vgg16_lower_part import vgg16_lower_part
from keras_bottleneck_train import display_stats

# path to the model weights files.
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
finetuned_weights = 'finetuned_weights.h5'
# dimensions of our images.
img_width, img_height = 300, 300

train_data_dir = 'data/random1000'#'data/random1000'
validation_data_dir = 'data/validation_A'#'data/aug_1024'
nb_train_samples = 4096
nb_validation_samples = 2944
nb_epoch = 50
dropout = 0.5
l2_c = 0.15

def load_model():
    # build the VGG16 network
    model = vgg16_lower_part(img_width, img_height, weights_path=weights_path)

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, W_regularizer=l2(l2_c), activation='relu'))
    top_model.add(Dropout(dropout))
    top_model.add(Dense(1, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    print(model.summary())
    return model

def train_model(model):
    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=32,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=32,
            class_mode='binary')

    cb = keras.callbacks.CSVLogger('log_epochs_finetuning.csv', separator=',', append=False)
    # fine-tune the model
    history = model.fit_generator(
            train_generator, callbacks =[cb],
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples)

    model.save_weights(finetuned_weights)
    return history

if __name__ == '__main__':
    model = load_model()
    history = train_model(model)
    display_stats(history)
