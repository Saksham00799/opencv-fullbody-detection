import cv2
import numpy as np
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml ')
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('TOKYO REVERSE OfficialTrailer (4K) Video of a Man Walking Backwards through Tokyo played in Reverse.mp4')
while True:
    ret,img=cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    body = fullbody_cascade.detectMultiScale(gray,1.3,3)
    for(x,y,w,h) in body:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        face=face_cascade.detectMultiScale(roi_gray,1.3,3)
        eyes=eye_cascade.detectMultiScale(roi_gray,1.3,3)
        smile=smile_cascade.detectMultiScale(roi_gray,1.3,3)
        for(fx,fy,fw,fh) in face :
            cv2.rectangle(roi_color,(fx,fy),(fx+fw,fy+fh),(0,0,255),2)
        for(ex,ey,ew,eh) in eyes :
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,125),2)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
            
    cv2.imshow('img',img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
