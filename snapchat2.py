import numpy as np
import cv2
import dlib
import math
hatt=cv2.imread('a.png',-1)
specss=cv2.imread('specs.png',-1)

def specs(w,h,frame,x,y):
    spec=cv2.resize(specss,(w,h))
    spec_mask=spec[:,:,3]
    spec_mask_inv=cv2.bitwise_not(spec_mask)
    
    spec=spec[:,:,0:3]
    only_spec=cv2.bitwise_and(spec,spec,mask=spec_mask)
    #cv2.imshow('h',only_hat)
    frame_roi=frame[y:y+h,x:x+w]
    #print(frame_roi.shape)  
    frame_roi=cv2.bitwise_and(frame_roi,frame_roi,mask=spec_mask_inv)
    #print(only_hat.shape)
    
    #cv2.imshow('s',frame_roi)
    merged=cv2.add(frame_roi,only_spec)
    frame[y:y+h,x:x+w]=merged
    return frame
def hat(w,h,frame,x,y):
    
    
    ha=cv2.resize(hatt,(w,h))
    ha_mask=ha[:,:,3]
    ha_mask_inv=cv2.bitwise_not(ha_mask)
    
    ha=ha[:,:,0:3]
    only_hat=cv2.bitwise_and(ha,ha,mask=ha_mask)
    #cv2.imshow('h',only_hat)
    frame_roi=frame[y:y+h,x:x+w]
    #print(frame_roi.shape)  
    frame_roi=cv2.bitwise_and(frame_roi,frame_roi,mask=ha_mask_inv)
    #print(only_hat.shape)
    
    #cv2.imshow('s',frame_roi)
    merged=cv2.add(frame_roi,only_hat)
    frame[y:y+h,x:x+w]=merged
    return frame
    
color_eyebrows=(255,255,255)
def calculatex(a,b,r,y):
    x=(r**2-(y-b)**2)
    if(x>0):
        x=x**0.5
        x=x+a
        x=int(x)
    return x
def calculatexx(a,b,r,y):
    x=(r**2-(y-b)**2)
    if(x>0):
        x=x**0.5
        x=-x+a
        x=int(x)
    return x
def tup(a,b,c,d):
    dlib_rect=dlib.rectangle(int(a),int(b),int(c),int(d))
    return dlib_rect

predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(1)
while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=fa.detectMultiScale(grey,1.1,10)
    for (x,y,w,h) in face:
        z=int((y+h/4)*0.97)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        d=tup(x,y,x+w,y+h)
        features=predictor(grey,d)
        hat_width=int(w+w*.25)
        hat_height=int(h*0.6)
        y=int(y-0.35*h)
        x=int(x-0.13*w)
        frame=hat(hat_width,hat_height,frame,x,y)
        frame=specs(w,(features.part(30).y-features.part(27).y)*2,frame,int(features.part(17).x-w*0.19),features.part(17).y-int(0.05*h))
        #if(b<z):
            #color_eyebrows=(0,0,255)
        #else:
            #color_eyebrows=(255,255,255)
        for i in range(1,68):
            font=cv2.FONT_HERSHEY_SIMPLEX
            st=str(i)
            #cv2.circle(frame,(features.part(i).x,features.part(i).y),3,(255,0,0),-2)
            #cv2.putText(frame,st,(features.part(i).x,features.part(i).y), font, 0.25,(255,255,255),1,cv2.LINE_AA)
            
            
            cv2.imshow('final',frame)
    if(cv2.waitKey(1)&0XFF==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
            

    
