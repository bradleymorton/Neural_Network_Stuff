#Created by Bradley Morton Nov 30

import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from keras.models import load_model
from PIL import Image, ImageEnhance
import numpy as np
import sys
from pathlib import Path


# dimensions of our images.
img_width, img_height = 150, 150

nb_train_samples = 1418
nb_validation_samples = 440
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

def createModel():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    return model

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

epochs = 10

finalVals =[]
for i in range(15):
    train_data_dir= "/home/bradley/College_Stuff/Year_5/Semester_1/CS_480/Neural_Network_Stuff/Generalized_Cross_Validation/case"+str(i)+"/train/"
    validation_data_dir = "/home/bradley/College_Stuff/Year_5/Semester_1/CS_480/Neural_Network_Stuff/Generalized_Cross_Validation/case"+str(i)+"/test/"
    final_validation_dir = "/home/bradley/College_Stuff/Year_5/Semester_1/CS_480/Neural_Network_Stuff/Generalized_Cross_Validation/case"+str(i)+"/validate/"

    train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

    intermediates = []
    for j in range(5):
        model = createModel()
        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

        count = 0
        total = 0

        for f in os.listdir(final_validation_dir):
            #print(f+" in case "+str(i)) #This was for debugging purposes- for some reason some of the images were not converting properly
            total+=1
            img = Image.open(final_validation_dir + f)
            img = img.resize((150,150))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,150,150,3)
        # Predicting the Test set results
            y_pred = model.predict_classes(im2arr)  
            flat_list = [item for sublist in y_pred for item in sublist]
    
            pred=flat_list[0]
            real=0
            if f[0]=='r':
                real=1
    
            if pred==real:
                count+=1    
          
        intermediates.append(count/total)    

    temp =0
    for k in range(len(intermediates)):
        temp+=intermediates[k]
    finalVals.append(temp/len(intermediates))

print(finalVals)


















