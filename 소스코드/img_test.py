from gluoncv import data, utils
from mxnet import gluon

import mxnet as mx
import cv2
import numpy as np
import cvlib

ctx = mx.cpu()


frame = 'C:\\image\\test\\1 (857).jpg'
img = cv2.imread(frame)

img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.add(img,80)
img=img.astype(np.float32)
img=((img-img.min())*(255)/(img.max()-img.min()))
img=img.astype(np.uint8)
img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    
x, img = data.transforms.presets.yolo.load_test(frame, short=415)
x = x.as_in_context(ctx)
net=gluon.SymbolBlock.imports(symbol_file='C:\\data\\darknet53-symbol.json', input_names=['data'], param_file='C:\\data\\darknet53-0000.params', ctx=ctx)



box_ids,scores, bboxes = net(x)
a=bboxes.asnumpy()
a=a.astype(int)
#print(box_ids)
#print(scores)
for i in range(0,len(box_ids[0])):
    if box_ids[0][i]!=-1 and scores[0][i]>0.5:
        left=a[0][i][0]
        top=a[0][i][1]
        right=a[0][i][2]
        bottom=a[0][i][3]
        if box_ids[0][i]==0:
            cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),1)
            cv2.putText(img,"safe",(left,top-10),
                       cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        else:
            cv2.rectangle(img,(left,top),(right,bottom),(255,0,0),1)
            cv2.putText(img,"unsafe",(left,top-10),
                       cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)

cv2.imshow('image', img[...,::-1])
cv2.waitKey(0)
cv2.imwrite("C:\\image\\result\\result_857.jpg", img[...,::-1])
cv2.destroyAllWindows()
