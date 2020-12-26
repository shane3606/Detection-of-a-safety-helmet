import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

video=cv2.VideoCapture("C:\\img\\test1.mp4")
video.set(cv2.CAP_PROP_FRAME_WIDTH,320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,240)


if not video.isOpened():
    print("Could not open video")
    exit()

while video.isOpened():
    ret, frame=video.read()
 
    if not ret:
        break
    bbox,label,conf=cv.detect_common_objects(frame)
    for i in range(len(label)):
        if label[i]=='person':
            a=(int)((bbox[i][3]-bbox[i][1])/5)
            b=(int)((bbox[i][2]-bbox[i][0])/4)
            left=bbox[i][0]+b
            top=bbox[i][1]
            right=bbox[i][2]-b
            bottom=bbox[i][1]+a
            cv2.rectangle(frame,(left,top),
                          (right,bottom),(0,255,0),1)
            img_roi=frame[top:bottom,left:right]
            m=cv2.mean(img_roi)
            if (m[0]+m[1]+m[2])/3<125:
                cv2.putText(frame,"unsafe",(left,top-10),
                       cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
            else:
                cv2.putText(frame,"safe",(left,top-10),
                       cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)
    #frame=cv2.resize(frame,None,None,0.5,0.5,cv2.INTER_AREA)
    cv2.imshow("img",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
