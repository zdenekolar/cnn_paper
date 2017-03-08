# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 23:02:53 2017

@author: User
"""

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import h5py
import numpy as np
from os import listdir
from shutil import copyfile
from sklearn.metrics import confusion_matrix
from vgg16_lower_part import vgg16_lower_part

weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
img_width = 300
img_height = 300

def initiate_models():
    '''
    Load the VGG-16 network.
    Load weights for the first part of the network.
    Create top layers for VGG-16
    Load weights for top weights
    :return:
    '''
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')
    
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
    
    model = initiate_models()
    pos = 0
    neg = 0
    
    path = r'D:\ConvNet\cnn_paper\data\validation_X\0/'
    miss = r'D:\ConvNet\cnn_paper\missclassified_bottleneck/'
    files = listdir(path)
    count = 0

    for i in files:#range(100,399):
        count += 1
        img = load_img(path + i)
        img = img.resize((300, 300))
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        x = x / 255
        
        test_datagen = ImageDataGenerator(rescale=1./255)
    #    test_generator = test_datagen.flow(x)
        
        prediction = model.predict_classes(x, batch_size=1)
        # confusion_matrix()
#        proba = model.predict_proba(x, batch_size=1)
#        print(prediction, proba)        
#        print (np.argmax(prediction))
        if prediction > 0.5:
            
            pos += 1
            copyfile(path + i, miss + str(count) + '.jpg')
            print('Pos')
        else:            
            neg += 1
            print('Neg')
            
            
            print(i)
        print(cat, dog)
