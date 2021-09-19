import cv2
import numpy as np
import utlis
import predict
##########################

webcam = True
scale =0.5#change scale when picture or camera changed
save_image_bool=False#determines which method of passing the image to prediction, by file or directly
remove_shadow = False
path_to_sample = 'cardboard_5.jpg'
cap = cv2.VideoCapture(2)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)

predict.prediction(cv2.imread(path_to_sample),save_image_bool)#initialises cudnn libraries

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path_to_sample)


    img = cv2.resize(img, (0, 0), None, scale, scale)
    img, conts =  utlis.getContours(img,remove_shadow,showCanny=True, draw= True,minArea=25000,filter=4)#change minArea if required

    coodinate = (0,80)
    area = 0

    if len(conts)!=0:
        area = conts[0][1]#change to 1 for area, 2 for rectangle sized object 4 coordinates
        
        # coodinate = conts[0][2][0][0]
        # print(conts[0][2][1][0])
        smallest_x=2000
        for point in conts[0][2]:
            # print(point[0][0])
            x,y=point[0][0],point[0][1]
            if x < smallest_x:#open array followed by x coordinate
                smallest_x = x
                coodinate = (x,y)
            # print(smallest_x)




    material = "Nil"

    if area>0:
        if save_image_bool:
            save_img_name = "opencv_frame_1.jpg"
            cv2.imwrite(save_img_name, img)
            material = predict.prediction("/" + save_img_name,save_image_bool)
        else:
            material = predict.prediction(img,save_image_bool)


    print_str = "Area: {area_display}, Material: {material_display}".format(area_display=area/1000,material_display=material)


    cv2.putText(img, print_str,
                (coodinate),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2)

    cv2.imshow('Original', img)
    cv2.waitKey(1)
