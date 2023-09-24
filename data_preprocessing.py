# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 18:44:44 2023

@author: youmn
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

#Data preprocessing

# Define your dataset path and categories
dataset_path=r"dataset"
categories=['with_mask','without_mask']

data=[]
labels=[]

# Loop through categories and load images
for category in categories:
   path=os.path.join(dataset_path, category)
   for img in os.listdir(path):
       img_path=os.path.join(path,img)
       image=load_img(img_path,target_size=(224,224))
       image=img_to_array(image)
       image=preprocess_input(image)
       data.append(image)
       labels.append(category)
       
       
#Encode the labels
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
#convert binarized label to one-hot encoded format
labels=to_categorical(labels)


data=np.array(data,dtype='float32')
labels=np.array(labels)

(trainX,testX, trainY,testY)=train_test_split(data,labels,test_size=0.20, stratify=labels,random_state=42)


learning_rate=0.0001
Epochs=20
batchSize=32

#genrate image for data augmentaion

aug=ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
    )

# Create a base neural network (CNN) model using MobilNetv

# Specify that we don't want to include the original classification layers ("top" layers) of MobileNetV2.
# We will add our own custom classification layers for our specific task later
baseModel=MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

#Consturct  fully connected layers on top of basemodel

#create new head for neural network based on the output of the base model
headModel=baseModel.output

#Apply average pooling to reduce spatial dimensions
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
#flatten the output to prepate it for fully connected layers
headModel=Flatten(name='flatten')(headModel)
headModel=Dense(128,activation='relu')(headModel)
#Apply Dropout regularization to prevent overfitting
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation='softmax')(headModel)

model=Model(inputs=baseModel.input , outputs=headModel)

#Freezing all layers in the base model to retain pre-trained knowledge while training only the head layers
for layer in baseModel.layers:
    layer.trainable=False
    
print(" model compiling...")

opt=LegacyAdam(lr=learning_rate, decay=learning_rate/Epochs)
model.compile(loss="binary_crossentropy",optimizer=opt)


#traing the head of the network

print( "training head ...")

hist = model.fit(
    aug.flow(trainX, trainY, batch_size=batchSize),
    steps_per_epoch=len(trainX),
    validation_data=(testX, testY),
    validation_steps=len(testX),
    epochs=Epochs
)


#Evaluation

print(" evaluting  network...")
predicion_probabilities=model.predict(testX,batch_size=batchSize)
#Determine the class labels with the higest predicted probabilities
ypred=np.argmax(predicion_probabilities,axis=1)

print(classification_report(testY.argmax(axis=1), ypred,target_names=lb.classes_))

#save the model
model.save("face_mask_detection.model",save_format='h5')




    