#hello everyone this is satinder singh from army institute of technology,pune
#i am trying to replicate the filters in snapchat
#i am a beginner right now and this is my first try
#the code is very simple and i think no one will have any problem understanding it
#I will be trying to increase the level to improve performance and visual effects

import cv2
import dlib


    
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
def eyebrows(features,frame,e):
    a=features.part(19).x
    b=features.part(19).y
    c=features.part(21).x
    d=features.part(21).y
    g=features.part(17).x
    h=features.part(17).y
    cv2.line(frame,(a,b),(c,d),color_eyebrows,10,cv2.LINE_AA)
    cv2.line(frame,(a,b),(g,h),color_eyebrows,10,cv2.LINE_AA)
    a=features.part(24).x
    b=features.part(24).y
    c=features.part(26).x
    d=features.part(26).y
    g=features.part(22).x
    h=features.part(22).y
    cv2.line(frame,(a,b),(c,d),color_eyebrows,10,cv2.LINE_AA)
    cv2.line(frame,(a,b),(g,h),color_eyebrows,10,cv2.LINE_AA)
    b=e-b
    return frame,b
def spectacles(features,frame):
            a=int(features.part(36).x+features.part(39).x)/2
            b=int(features.part(36).y)
            cv2.circle(frame,(int(a),int(b)),int(h/8.0),(234,145,0),6,cv2.LINE_AA)
            c=int(features.part(42).x+features.part(45).x)/2
            d=int(features.part(42).y)
            e=int(features.part(28).y)-10
            s=calculatex(a,b,h/8,e)
            f=calculatexx(c,d,h/8,e)
            f=int(f)
            cv2.circle(frame,(int(s),int(e)),5,(155,87,1),-2,cv2.LINE_AA)
            cv2.circle(frame,(int(f),int(e)),5,(155,87,1),-2,cv2.LINE_AA)
            cv2.line(frame,(int(s),int(e)),(int(f),int(e)),(0,234,174),4,cv2.LINE_AA)
            cv2.circle(frame,(int(c),int(d)),int(h/8.0),(234,145,0),6,cv2.LINE_AA)
            return frame,d
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
        frame,e=spectacles(features,frame)
        frame,b=eyebrows(features,frame,e)
        
        
        if(b>int(h/5)):
            color_eyebrows=(0,0,255)
        else:
            color_eyebrows=(255,255,255)
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
            

    
