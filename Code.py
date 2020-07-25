# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 15:34:00 2020

@author: HP
"""


import numpy as np
import cv2



face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

mouth_cascade= cv2.CascadeClassifier('mouth.xml')

#capturinng the video
cap=cv2.VideoCapture(0)
while 1:
    
    _,frame=cap.read()
    
    frame=cv2.flip(frame,1)
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #Thresholding to notice for white mask(on face)
    
    _,img1 = cv2.threshold(gray, 80, 255,cv2.THRESH_OTSU) 
    
    thresh = face_cascade.detectMultiScale(img1, 1.3, 5)
    
    faces=face_cascade.detectMultiScale(gray,1.3,10)
    
    if (len(faces)==0 and len(thresh) == 0):
       cv2.putText(frame,'Face not there', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    elif(len(faces) == 0 and len(thresh) == 1):
        # This is the case of wearing a white mask ,
        #where the face recognition does not takes place
        #in case of a gray image and takes place in case of the thresholded image
            cv2.putText(frame,'wearing mask', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    else:
        
       #simple face detection logic where we find coordinates on the face
       #
        
        for (x,y,w,h) in faces:
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
            roi_gray = gray[y:y+h, x:x+w]
            
            roi_color = frame[y:y+h, x:x+w]

            mouth=mouth_cascade.detectMultiScale(gray,1.4,5)
         
        if (len(mouth)==0):
            
            cv2.putText(frame,'wearing mask', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        
        else:
            #here we are checking that muth coordinates are inside the face
            #which is always true which further means that mouth is being detected
            #and the person is not wearing mask
        
            for (mx, my, mw, mh) in mouth:

                if(y < my < y + h):

                      cv2.putText(frame,'not wearing mask', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow('frame',frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()    
cap.release