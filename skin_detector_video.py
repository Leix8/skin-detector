import numpy as np
import argparse
import cv2

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v","--video",help="path to the optional video file")
    args = vars (ap.parse_args())
    args = vars(ap.parse_args())

    lower = np.array( [ 0,48,80], dtype= "uint8")
    upper = np.array( [ 20,255,255 ], dtype= "uint8")
    
    if not args.get( "video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])

    while True:
        (grabbed, frame)= camera.read( )
        if args .get ("video") and not grabbed:
            break
        frame=cv2.resize(frame,(600,400))
        converted =cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask=cv2.inRange(converted, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        skinMask=cv2.erode(skinMask,kernel,iterations=2)
        skinMask=cv2.dilate(skinMask,kernel,iterations=2)

        skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
        skin=cv2.bitwise_and(frame,frame,mask=skinMask)

        cv2.imshow("images",np.hstack([frame,skin]))

        if cv2.waitKey(1)&0xFF ==ord("q"):
            break

    camera.release( )
    cv2.destroyAllWindows()
    





