import cv2
import numpy as np

def removeShadows(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    return result_norm


def getContours(img,remove_shadow,cThr = [40, 150], showCanny = False, minArea = 1000,filter = 0, draw=False):#if diff lighting, change canny edge threshold. 40,150
    #Note for lower and upper threshold
    #1) Make both values the same
    #2) Decrease untill all the edges that you want can be seen
    #3) Increase the right (max) threshold untill the edges you don't want dissapear

    if remove_shadow:
        imgGray = cv2.cvtColor(removeShadows(img), cv2.COLOR_BGR2GRAY)
    else:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)


    if showCanny:
        cv2.imshow('Canny', imgThre)

    contours, hierarchy = cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    finalContours = []
    for i in contours:
        area = cv2.contourArea(i);
        if area > minArea:
            per=cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i, 0.02*per, True)#if detecting another shape, this must change
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx), area, approx, bbox,i])
            else:
                finalContours.append([len(approx), area, approx, bbox,i])

    finalContours = sorted(finalContours, key = lambda x:x[1], reverse = True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img,con[4], -1, (0, 0, 255),3)

    return img, finalContours;

def warpImg(img, points, w,h):
    print("hello")