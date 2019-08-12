import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fullbody = fullbody_cascade.detectMultiScale(gray,1.3,3)
    for (x,y,w,h) in fullbody:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        faces = face_cascade.detectMultiScale(roi_gray,1.3,5)
        for (fx,fy,fw,fh) in faces:
            cv2.rectangle(roi_color,(fx,fy),(fx+fw,fy+fh),(0,255,0),2)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
        
                
    cv2.imshow('img',img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
