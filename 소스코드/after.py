import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import numpy as np

video=cv2.VideoCapture("C:\\img\\test1.mp4")


if not video.isOpened():
    print("Could not open video")
    exit()

while video.isOpened():
    ret, frame=video.read()
    if not ret:
        break

    #histograme을 분석하고 정규화 하는 과정
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame=cv2.add(frame,80)
    frame=frame.astype(np.float32)
    frame=((frame-frame.min())*(255)/(frame.max()-frame.min()))
    frame=frame.astype(np.uint8)
    frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    
    #객체를 검출하는 과정 
    bbox,label,conf=cv.detect_common_objects(frame)
    for i in range(len(label)):
        if label[i]=='person':
            #머리 부분을 가져오는 과정
            a=(int)((bbox[i][3]-bbox[i][1])/5)
            b=(int)((bbox[i][2]-bbox[i][0])/4)
            left=bbox[i][0]+b
            top=bbox[i][1]
            right=bbox[i][2]-b
            bottom=bbox[i][1]+a
            cv2.rectangle(frame,(left,top),
                          (right,bottom),(0,255,0),1)
            
            #머리 부분을 이진화하는 과정
            img_roi=frame[top:bottom,left:right]
            img_roi=cv2.cvtColor(img_roi,cv2.COLOR_BGR2GRAY)
            _,img_roi=cv2.threshold(img_roi,80,255,cv2.THRESH_BINARY)
            img_roi=cv2.cvtColor(img_roi,cv2.COLOR_GRAY2BGR)
            m=cv2.mean(img_roi)

            #안전모를 검출하는 과정
            if (m[0]+m[1]+m[2])/3<125:
                cv2.putText(frame,"unsafe",(left,top-10),
                       cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
            else:
                cv2.putText(frame,"safe",(left,top-10),
                       cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)

    cv2.imshow("img",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
