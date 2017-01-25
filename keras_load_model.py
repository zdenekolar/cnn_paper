# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:43:29 2017

@author: User
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import model_from_json
import numpy as np

with open('config.json', 'r') as f:
    json_string = f.read()

model = model_from_json(json_string)
model.load_weights('weights.h5')

print(model.summary())

for i in range(10,30):
    img = load_img(r'C:\Git\version-control\cnn_paper\datasets\accordion/image_00{}.jpg'.format(i))
    img = img.resize((300, 300))
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    
    prediction = model.predict(x, batch_size=1)
#    print (np.argmax(prediction))
    print (prediction)