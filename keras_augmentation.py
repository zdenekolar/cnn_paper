# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:16:22 2017

@author: User
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
number_samples = 5000

datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='reflect',
        )

i = 0
for batch in datagen.flow_from_directory(
        r'C:\Users\User\Google Drive\PhD\Papers\Guardrail\images\test3',
        batch_size=1,
        target_size=(300, 300),        
        class_mode='binary',
        save_to_dir=r'C:\Users\User\Google Drive\Source_Codes\MNIST\test', 
        save_prefix='cat', 
        save_format='jpeg'):

    i += 1
    if i > number_samples:
        break  # otherwise the generator would loop indefinitely        