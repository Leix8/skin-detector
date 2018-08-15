import sys
import numpy as np
import argparse
import cv2
import colorsys
import matplotlib.pyplot as plt

#cascaded face detector
def detect_face(im):
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces=face_cascade.detectMultiScale(im,1.2,5)
    print("faces: ",faces)
    if faces ==():
        print("cannot detect face")
        sys.exit()
    return faces

#extract skin patches, input a BGR 255 image, return masked BGR 255
def skin_detect(im):
    face=detect_face(im)

    mask_init=np.zeros((im.shape[0],im.shape[1],im.shape[2]),dtype="uint8")

#    mask_face=cv2.rectangle(mask_init, (face[0,0],face[0,1]),((face[0,2]+face[0,0]),(face[0,1]+face[0,3])),color=200,thickness=1)
#    cv2.imshow("face mask: ",mask_face)
#    im_faceonly=cv2.bitwise_and(im,im,mask=mask_face)
    mask_init[face[0,1]:(face[0,1]+face[0,3]),face[0,0]:(face[0,0]+face[0,2])]=im[face[0,1]:(face[0,1]+face[0,3]),face[0,0]:(face[0,0]+face[0,2])]
    cv2.imshow("face only: ", mask_init)
    lower = np.array( [ 0,40,80], dtype= "uint8")
    upper = np.array( [ 20,255,255 ], dtype= "uint8")

    converted =cv2.cvtColor(mask_init, cv2.COLOR_BGR2HSV)
    skinMask=cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    skinMask=cv2.erode(skinMask,kernel,iterations=4)
    skinMask=cv2.dilate(skinMask,kernel,iterations=1)

    skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
    skin_bgr=cv2.bitwise_and(im,im,mask=skinMask)

    return(skin_bgr)

#screen filter, to brighted the image, need to be normalized
#def screen(im):
#    im_filtered=2*im-im*im
#    return im_filtered

#combine with screen filter
#def add_screen(im,skin):
#    ret , mask=cv2.threshold(skin,10,255,cv2.THRESH_BINARY)
#    mask_inv=cv2.bitwise_not(mask)
#    print(type(skin))
#    skin_patch=cv2.bitwise_and(skin,skin,mask=mask)
#    image=cv2.add(im,skin_patch)
#    return(image)

#take average of skin patches, as the representative color of skin in SHV space
def represent_color(skin):
    cnt=0
    sum=[0.0,0.0,0.0]
#    print("skin shape: ",skin.shape)
    for i in range(skin.shape[0]):
        for j in range(skin.shape[1]):
            if not (skin[i,j]==[0,0,0]).all():      #get rid of background pixels
#                if i==j:
#                    print("sample pixel: ",skin[i,j])
                sum=sum+skin[i,j]
                cnt+=1
#    print("sum and cnt: ",sum,cnt)
    avg=sum/cnt
    return (avg)

#
def sample_pixel(skin):
    pixels=[]
    for k in range(10):
        for i in range(skin.shape[0]):
            for j in range(skin.shape[1]):
                if skin[i,j].all!=0:
                    pixels.append(skin[i,j])
    return pixels


def to_hsv( color ):
    """ converts color tuples to floats and then to hsv """
    color=colorsys.rgb_to_hsv(*[x/255.0 for x in color])
#    print("color in hsv: ",color)
    return (color) #rgb_to_hsv wants floats!

#compute color distance in hsv colorspace
def color_dist( c1, c2):
    """ returns the squared euklidian distance between two color vectors in hsv space """
    return sum( (a-b)**2 for a,b in zip(to_hsv(c1),to_hsv(c2)) )

#find the closest color with minimum color distance
#avg is the representative color and colors are the set of color samples
def min_color_diff( avg, colors):
    """ returns the `(distance, color_name)` with the minimal distance to `colors`"""
    return min( # overal best is the best match to any color:
        (color_dist(avg, test), colors[test]) # (distance to `test` color, color name)
        for test in colors)


if __name__=="__main__":

    im=cv2.imread('0.jpg')
    cv2.imshow('original',im)
    colors = dict((
    ((233,210,214), "1"),
    ((231,210,194), "2"),
    ((240,241,191), "3"),
    ((221,182,157), "4"),
    ((225,189,160), "5"),
    ((182,137,108), "6"),
    ((167,121,96), "7"),
    ((150,96,58), "8"),
    ((136,79,54),"9"),
    ((61,44,40),"10"))) #in 255 RGB

    skin_bgr=skin_detect(im) # 255 HSV
    cv2.imshow('maksed',skin_bgr)


    avg_bgr=represent_color(skin_bgr) #01 HSV

    print("avge color in bgr: ",avg_bgr)
    avg_rgb=np.flip(avg_bgr,0)
    min_dist, rank=min_color_diff(avg_rgb,colors)
    print(rank)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#    im=cv2.normalize(im,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F) #0-1BGR
#    skin=cv2.normalize(skin,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)#01 HSV

#    skin_screen=screen(skin) #?
##    brightness=bright_eval(skin)
##    print(brightness)
#    cv2.imshow("screen",skin_screen)
#    cv2.waitKey(0)


#    color_show=255*np.asarray(colorsys.hsv_to_rgb(avg[0],avg[1],avg[2]))
#    print("avg in RGB 255: ",color_show)
#    avg=np.asarray(avg)

#    plt.imshow([avg_rgb])
#    plt.show()
#    print(skin.shape[0])
#    im_add=add_screen(im,skin)

#    cv2.imshow("overlay",im_add)
#    cv2.waitKey(0)



    





