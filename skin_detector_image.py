import numpy as np
import argparse
import cv2

def skin_detect(im):

    lower = np.array( [ 0,40,80], dtype= "uint8")
    upper = np.array( [ 20,255,255 ], dtype= "uint8")

    converted =cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    skinMask=cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    skinMask=cv2.erode(skinMask,kernel,iterations=2)
    skinMask=cv2.dilate(skinMask,kernel,iterations=2)

    skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
    skin=cv2.bitwise_and(im,im,mask=skinMask)

    return(skin)

def screen(im):
    im_filtered=2*im-im*im
    return im_filtered

#def bright_eval(skin):
#    cnt=0
#    sum=0
#    for p in skin:
#        if p!=0:
#            sum=sum+p
#            cnt+=1
#    avg=sum/cnt
#    return (avg)
def add_screen(im,skin):
    ret , mask=cv2.threshold(skin,10,255,cv2.THRESH_BINARY)
    mask_inv=cv2.bitwise_not(mask)
    print(type(skin))
    skin_patch=cv2.bitwise_and(skin,skin,mask=mask)
    image=cv2.add(im,skin_patch)
    return(image)

if __name__=="__main__":

    

    im=cv2.imread('2.jpeg')

    cv2.imshow('original',im)
    cv2.waitKey(0)
    skin=skin_detect(im)
    cv2.imshow('maksed',skin)
    cv2.waitKey(0)
    im=cv2.normalize(im,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    skin=cv2.normalize(skin,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    skin_screen=screen(skin)
#    brightness=bright_eval(skin)
#    print(brightness)
    cv2.imshow("screen",skin_screen)
    cv2.waitKey(0)

    im_add=add_screen(im,skin)

    cv2.imshow("overlay",im_add)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    





