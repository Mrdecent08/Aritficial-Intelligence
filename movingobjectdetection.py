import cv2
import time
import imutils
cam = cv2.VideoCapture(1)
time.sleep(1)
firstFrame = None
area = 500
while True:
    _,img = cam.read()
    text = 'normal'
    img = imutils.resize(img,width=500)
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg,(21,21),0)
    if firstFrame is None:
        firstFrame = gaussianImg
        continue
    imdDiff = cv2.absdiff(firstFrame,grayImg)
    thresImg = cv2.threshold(imdDiff ,25,255,cv2.THRESH_BINARY)[1]
    thresImg = cv2.dilate(thresImg,None,iterations=2) #borders and holes removal
    cnts = cv2.findContours(thresImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)#connectting neighbourhood pixels
    for c in cnts:
        if cv2.contourArea(c)<area:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        text = "moving object detected"
    print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow("cameraFeed",img)
    key = cv2.waitKey(1)&0xFF
    if key == ord("q"):
        break
cam.release()