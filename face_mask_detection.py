# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:02:25 2023

@author: youmn
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import imutils 
from imutils.video import VideoStream
import numpy as np
import os
import cv2
import time

#load pretrained model for face detection

#model architecture file path
model_config_path =r'face_detector\deploy.prototxt'
#pre trained weights file 
model_weights_path =r'face_detector\res10_300x300_ssd_iter_140000.caffemodel'
face_detection_model =cv2.dnn.readNet(model_config_path,model_weights_path)

#load face mask detection (our model)
face_mask_detection_model=load_model("face_mask_detection.model")

#initilize the video stream
print("starting video stream....")
vs=VideoStream(src=0).start()

def detect_and_predict_mask(frame,facemodel,maskmodel):
    #Get the dimensions of the frame
    (img_height,img_width)=frame.shape[:2]
    #Create a preprocessed image from the given frame
    preprocessed_frame=cv2.dnn.blobFromImage(frame,1.0,(224,224),(104.0,177.0,123.0))
    
    face_detection_model.setInput(preprocessed_frame)
    face_detections=face_detection_model.forward()
    # Print the shape of the face detections (number of detected faces and their attributes).
    print("Face Detections Shape:", face_detections.shape)
    
    faces=[]
    locations=[]
    predictions=[]
    
    #Iterate through objects that are detected
    for i in range(0,face_detections.shape[2]):
       #Retrieving the confidence score associated with the i-th detection
       confidence_score=face_detections[0,0,i,2]
       if confidence_score>0.5:
          #Extract the bounding box coordinate from face_detection array
          bounding_box=face_detections[0,0,i,3:7]
          #scale the bounding box
          bounding_box=bounding_box*np.array([img_width,img_height,img_width,img_height])
          (startX,startY,endX,endY)=bounding_box.astype(int)
       
          #Clipping bounding box coordinates
          (startX,startY)=(max(0,startX),max(0,startY))
          (endX,endY)=(min(img_width-1,endX),min(img_height-1,endY))
       
          #Extract the face using bounding box coordinate
          face=frame[startY:endY,startX:endX]
          face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
          face=cv2.resize(face,(224,224))
          face=img_to_array(face)
          face=preprocess_input(face)
          faces.append(face)
          locations.append((startX,startY,endX,endY))
       
    #at least one face is detected   
    if len(faces)>0:
        faces=np.array(faces,dtype='float32')
        predictions=face_mask_detection_model.predict(faces,batch_size=32)
        
        
    return (locations,predictions)
       
       
while True:
    #Captures a frame from the video stream and resizes it
    frame=vs.read()
    frame=imutils.resize(frame,width=400)
    
    #detect face and determine if he is wearing the mask or not
    (face_locations, mask_predictions)=detect_and_predict_mask(frame,face_detection_model,face_mask_detection_model)
    
    #Iterates over bounding box and croresponding mask predicion for each detected faces 
    
    for (box,pred) in zip(face_locations, mask_predictions):
        #getting bounding box coordinates
        (startX,startY,endX,endY)=box
        #unpack predicions into the probabiility of wearing mask or not
        (mask,without_mask)=pred
        label="Mask" if mask>without_mask else "No Mask"
        #make the color of bounding box green in case wearing mask if not make the color red
        color=(0,255,0) if label=='Mask' else (0,0,255)        
        #show the probability of the predicated label
        label='{}: {:2f}%'.format(label,max(mask,without_mask)*100)
        #show the bounding box and crossponding information
        cv2.putText(frame,label,(startX,startY-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2
                    )
        cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
    
    #show the output frame
    cv2.imshow('Frame',frame)
    #wait for a key press (1 ms timeout)
    key = cv2.waitKey(1) & 0xFF
    #if key 's' stop and break the loop
    if key==ord('s'):
        break
   
    
cv2.destroyAllWindows()
vs.stop()  
    
      

  
    