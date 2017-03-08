# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 20:19:28 2017

@author: User
"""

from vgg_16_keras import VGG_16
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import model_from_json
from os import listdir
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l1, l2
from vgg16_lower_part import vgg16_lower_part
img_width, img_height = 300, 300

finetuned_weights = 'finetuned_weights.h5'
dropout = 0.5
l2_c = 0.15

def model(weights_path):
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
    
    model = model(finetuned_weights)
    # TODO: Implement proper statistics, using scikit learn
    pos = 0
    neg = 0


paths = [r'D:\ConvNet\cnn_paper\data\validation_X\1/']#,r'D:\ConvNet\cnn_paper\data\validation_X\0']

for p in paths:
    files = listdir(p)
    for i in files:
        img = load_img(r'{}/{}'.format(p,i))
        img = img.resize((300, 300))
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        x = x / 255

        test_datagen = ImageDataGenerator(rescale=1./255)
    #    test_generator = test_datagen.flow(x)

        prediction = model.predict_classes(x, batch_size=1)
        # proba = model.predict_proba(x, batch_size=1)
        # print(prediction, proba)
#        print (np.argmax(prediction))
        if prediction > 0.5:

            pos += 1
            print('pos')
        else:
            neg += 1
            print('neg')
        print(pos, neg)

