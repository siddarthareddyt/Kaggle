#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 00:40:56 2017

@author: siddartha
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout

from timeit import default_timer as timer

shape_w = 150
shape_h = 150
batchsize = 32

start = timer()


classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(shape_w, shape_h, 3)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=64, activation="relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1, activation="sigmoid"))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (shape_w, shape_h),
                                                 batch_size = batchsize,
                                                 class_mode = 'binary')

validation_set = validation_datagen.flow_from_directory('dataset/validation_set',
                                            target_size = (shape_w, shape_h),
                                            batch_size = batchsize,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = (10000 // batchsize),
                         epochs = 50,
                         validation_data = validation_set,
                         validation_steps = (2500 // batchsize))
classifier.save('catsdogs.h5')
classifier.save_weights('catsdogs_weights.h5')

end = timer()
print(end - start)
import os
os.system('say "your program has finished"')


# predict on new images
basedir = 'test/'
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd

rows = []
for i in range(0,12494):
    
    path = basedir + str(i) + '.jpg'
    img = load_img(path,False,target_size=(shape_w,shape_h))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = classifier.predict_classes(x)
    probs = classifier.predict_proba(x)
    row = {'id': i, 'label': preds[0][0]}
    rows.append(row)
    print(preds)
    print(probs)
    print("-----------------")
    
predictions = pd.DataFrame(rows)
predictions.to_csv(path_or_buf="Submission2.csv", index=False, header=True)
    
#on a single image

path = basedir + str(4) + '.jpg'
img = load_img(path,False,target_size=(shape_w,shape_h))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = classifier.predict_classes(x)
probs = classifier.predict_proba(x)


