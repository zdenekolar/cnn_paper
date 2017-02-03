# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 20:19:28 2017

@author: User
"""

from vgg_16_keras import VGG_16
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import model_from_json

finetuned_weights = 'finetuned_weights.h5'

if __name__ == '__main__':
    
    model = VGG_16(finetuned_weights)
    cat = 0
    dog = 0

    for i in range(100,399):
        img = load_img(r'C:\Git\version-control\cnn_paper\validate\cat/cat.1{}.jpg'.format(i))
        img = img.resize((150, 150))
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        x = x / 255
        
        test_datagen = ImageDataGenerator(rescale=1./255)
    #    test_generator = test_datagen.flow(x)
        
        prediction = model.predict_classes(x, batch_size=1)
        proba = model.predict_proba(x, batch_size=1)
        print(prediction, proba)        
#        print (np.argmax(prediction))
        if prediction > 0.5:
            
            dog += 1
            print('Dog')
        else:            
            cat += 1    
            print('Cat')
        print(cat, dog)
