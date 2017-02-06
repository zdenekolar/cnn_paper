# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 07:56:27 2017

@author: Zdenek
"""

import os
import random
import numpy as np
import h5py
import json
from matplotlib.pyplot import imshow
import matplotlib.pyplot
from PIL import Image
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.manifold import TSNE

vgg_path = r'vgg16_weights.h5'
images_path = r'D:\ConvNet\cnn_paper\data\train - small\1'
num_images = 500

def get_image(path):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)  # resize the image to fit into VGG-16
    img = np.array(img.getdata(), np.uint8)
    img = img.reshape(224, 224, 3).astype(np.float32)
    img[:,:,0] -= 123.68 # subtract mean (probably unnecessary for t-SNE but good practice)
    img[:,:,1] -= 116.779
    img[:,:,2] -= 103.939
    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0)
    return img
    
def VGG_16(weights_path):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    print("finished loading VGGNet")
    return model
    
model = VGG_16(vgg_path)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
if num_images < len(images):
    images = [images[i] for i in sorted(random.sample(range(len(images)), num_images))]

print("keeping %d images to analyze" % len(images))

activations = []
for idx,image_path in enumerate(images):
    if idx%100==0:
        print ("getting activations for %d/%d %s" % (idx+1, len(images), image_path))
    image = get_image(image_path);
    acts = model.predict(image)[0]
    activations.append(acts)
    
matplotlib.pyplot.plot(np.array(activations[0]))
matplotlib.pyplot.show()

# first run our activations through PCA to get the activations down to 300 dimensions
activations = np.array(activations)
pca = PCA(n_components=300)
pca.fit(activations)
pca_activations = pca.transform(activations)

# then run the PCA-projected activations through t-SNE to get our final embedding
X = np.array(pca_activations)
tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, verbose=2, angle=0.2).fit_transform(X)

print('t-sne finished')
# normalize t-sne points to {0,1}
tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

width = 3000
height = 3000
max_dim = 100

print('creating image')

count=0
full_image = Image.new('RGB', (width, height))
for img, x, y in zip(images, tx, ty):
    print(count)
    count+=1
    
    tile = Image.open(img)
    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)))

matplotlib.pyplot.figure(figsize = (12,12))
imshow(full_image)

full_image.save("myTSNE.png")