# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 20:19:28 2017

@author: User
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import model_from_json
from os import listdir
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l1, l2
from vgg16_lower_part import vgg16_lower_part
from validation import testing_stas
img_width, img_height = 300, 300

finetuned_weights = 'finetuned_weights.h5'
dropout = 0.5
l2_c = 0.15

def initialize_model(weights_path):
    '''
    Load lower VGG16 model and top VGG16 model.
    Merge the two models to get a complete model.
    Load weights.
    :param weights_path:
    :return:
    '''

    model = vgg16_lower_part(img_width, img_height)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, W_regularizer=l2(l2_c), activation='relu'))
    top_model.add(Dropout(dropout))
    top_model.add(Dense(1, activation='sigmoid'))

    model.add(top_model)

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == '__main__':
    
    model = initialize_model(finetuned_weights)
    # TODO: Implement proper statistics, using scikit learn

    pos_data_dir = r'data\augmentation1000\1'
    neg_data_dir = r'data\augmentation1000\0'

    testing_stas(model, pos_data_dir, neg_data_dir)


