# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 23:02:53 2017

@author: User
"""

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D
from validation import testing_stas
import numpy as np

from vgg16_lower_part import vgg16_lower_part


weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'

def initiate_models():
    '''
    Load the VGG-16 network.
    Load weights for the first part of the network.
    Create top layers for VGG-16.
    Load weights for top weights.
    :return:
    '''

    model = vgg16_lower_part(300, 300, weights_path)

    train_data = np.load(('bottleneck_features_train.npy'))
    top_model = Sequential()
    top_model.add(Flatten(input_shape=train_data.shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    top_model.load_weights(top_model_weights_path)
    
    top_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # add the model on top of the convolutional base
    model.add(top_model)
    
    return model


if __name__ == '__main__':
    '''
    Entry point.
    '''

    pos_data_dir = r'data\augmentation1000\1'
    neg_data_dir = r'data\augmentation1000\0'

    model = initiate_models()
    testing_stas(model, pos_data_dir, neg_data_dir)

