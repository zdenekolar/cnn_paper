# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:16:22 2017

@author: User
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
number_samples = 50000

datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect',
        )

i = 0
for batch in datagen.flow_from_directory(
        r'D:\ConvNet\cnn_paper\data\augmentation1000',
        batch_size=1,
        target_size=(300, 300),        
        class_mode='binary',
        save_to_dir=r'D:\ConvNet\cnn_paper\data\augmented1000', 
        save_prefix='guardrail', 
        save_format='jpeg'):

    i += 1
    if i > number_samples:
        break  # otherwise the generator would loop indefinitely        