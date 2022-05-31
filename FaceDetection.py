import cv2
import argparse
import ntpath
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import imutils
from imutils.video import WebcamVideoStream
from imutils import face_utils
from Detect_Acne import *
import math

import time

import glob
import os


import dlib
predictor_model = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()# dlib人脸检测器
predictor = dlib.shape_predictor(predictor_model)

heightResize = 480
framesSkipping = 1

cameraObject = cv2.VideoCapture(0)
ret, image = cameraObject.read()
height = image.shape[0]

frame_resize_scale = float(height)/heightResize
modelPath = "shape_predictor_81_face_landmarks.dat"

faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(modelPath)



def readFiles(path):
    img = cv2.imread(path)
    return img



def img2Gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def showImage(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()



def GetFaceMask(img):
    img, Gimg = img2Gray(img)
    rects = detect(Gimg) # return multi-face
    
    # Face mask, cut out of face region
    Facemask = np.zeros_like(img)
    (y, x, w, h) = rects[0].astype("int")
    Facemask = cv2.rectangle(Facemask, (y,x), (y + w, x + h), (255,255,255), -1)
    
    return Facemask



def skinDetection(img):
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)
    
    return global_result


def get_landmarks(Gimg):

    detector = dlib.get_frontal_face_detector()
    
    StartTime = time.time() # start time
    predictor = dlib.shape_predictor(predictor_model)
    
    print(time.time() - StartTime) # end time
    
    rects = detector(Gimg, 0)
    face = rects[0]
    
    shape = predictor(Gimg, face)
    shape = face_utils.shape_to_np(shape)

    
    landmarks = np.matrix([[p.x, p.y] for p in predictor(Gimg, face).parts()])
    for points in predictor(Gimg, face).parts():
        cv2.circle(Gimg, (points.x, points.y), 1, (0, 0, 255), 10)

    
    return face, landmarks


def getForeHead(landsmark):
    forehead = np.zeros([5,2], dtype = np.int)
    int_lmrks = np.array(landsmark, dtype=np.int)
    D_nose = abs(int_lmrks[27]-int_lmrks[30])
    print("D_nose : " + str(D_nose))
    L_eye = abs(int_lmrks[36][0] - int_lmrks[40][0])
    R_eye = abs(int_lmrks[42][0] - int_lmrks[45][0]) 
    
    leftEye = ((int_lmrks[36] + int_lmrks[39]) / 2)
    rightEye = ((int_lmrks[42] + int_lmrks[45]) / 2)
    centroidEye = ( (leftEye+rightEye) / 2)
    lineX = leftEye[0] - rightEye[0]
    lineY = leftEye[1] - rightEye[1]
    mag =  math.sqrt((lineX*lineX) + (lineY*lineY))
    
    lineX = lineX/mag
    lineY = lineY/mag
    
    VlineY = lineX
    VlineX = -lineY
    
    # diff from centroidEye to noise
    diff = (centroidEye - int_lmrks[30])
    length = math.sqrt((diff[0]*diff[0]+diff[1]*diff[1]))
    
    # diff from centroidEye to leftEye
    diffL = (centroidEye - int_lmrks[16])
    print("diffL" + str(diffL))
    lengthL = math.sqrt((diffL[0]*diffL[0]+diffL[1]*diffL[1]))

    # diff from centroidEye to rightEye
    diffR = (centroidEye - int_lmrks[0])
    print("diffR" + str(diffR))
    lengthR = math.sqrt((diffR[0]*diffR[0]+diffR[1]*diffR[1]))

    
    # normalize C

#     print("test")
#     print(int_lmrks[5][0])
#     print(int_lmrks[17])
#     print(int_lmrks[11])
#     print(int_lmrks[26])
#     print("test")
    C = 1
    # a 
    forehead[0] =  int_lmrks[0]

    # b
    forehead[1][0] =  round(centroidEye[0] + 2*length * VlineX + 0.8*lengthR * lineX)
    forehead[1][1] =  round(centroidEye[1] + 2*length * VlineY + 0.8*lengthR * lineY)
    print("lengthR : " + str(lengthR))
    
    # c
    forehead[2] =  int_lmrks[16]
    
    # d
    forehead[3][0] =  round(centroidEye[0] + 2*length * VlineX - 0.8*lengthL * lineX)
    forehead[3][1] =  round(centroidEye[1] + 2*length * VlineY - 0.8*lengthL * lineY)
    print("lengthL : " + str(lengthL))
    
    # e
    forehead[4][0] =  round(centroidEye[0] + 2.5*length * VlineX)
    forehead[4][1] =  round(centroidEye[1] + 2.5*length * VlineY)
    
    return forehead





# Origin_img = cv2.imread('data/t4.jpg')
# img, Gimg = img2Gray(Origin_img)
# face, landmarks = get_landmarks(Gimg)
# result = getForeHead(landmarks)
# print(result.shape)
# for points in result:
#     print(points)
#     cv2.circle(img, (points[0], points[1]), 20, (0, 0, 255), 10)
    
# int_lmrks = np.array(landmarks, dtype=np.int)
# showImage(img)




def GetSkinMask(img):
    # find skin region, 0 is skin, otherwise is 1
    SkinImage = skinDetection(img)
    skinMask = np.zeros_like(img)
    skinMask[SkinImage[:,:] == 0] = (255,255,255)
    skinMask[SkinImage[:,:] == 1] = (0, 0, 0)
    
    return skinMask



#get_ipython().run_line_magic('pip', 'install pyclipper')
import pyclipper

def perimeter(poly):
    p = 0
    nums = poly.shape[0]
    for i in range(nums):
        p += abs(np.linalg.norm(poly[i % nums] - poly[(i + 1) % nums]))
    return p

def proportional_zoom_contour(contour, ratio):

    poly = contour[:, :]
    area_poly = abs(pyclipper.Area(poly))
    perimeter_poly = perimeter(poly)
    poly_s = []
    pco = pyclipper.PyclipperOffset()
    pco.MiterLimit = 10
    if perimeter_poly:
        d = area_poly * (1 - ratio * ratio) / perimeter_poly
        pco.AddPath(poly, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        poly_s = pco.Execute(-d)
    poly_s = np.array(poly_s).reshape(-1, 1, 2).astype(int)

    return poly_s

def get_image_hull_mask(img, image_landmarks, ie_polys=None):

    int_lmrks = np.array(image_landmarks, dtype=np.int)
    #hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)
    hull_mask = np.zeros_like(img)

    
    forehead = getForeHead(image_landmarks)
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(forehead), (255,255,255))
    
    
    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[0:9],
                        int_lmrks[17:18]))), (255,255, 255))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[8:17],
                        int_lmrks[26:27]))), (255,255, 255))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:20],
                        int_lmrks[8:9]))), (255,255, 255))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[24:27],
                        int_lmrks[8:9]))), (255,255, 255))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[19:25],
                        int_lmrks[8:9],
                        ))), (255,255, 255))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:22],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (255,255, 255))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[22:27],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (255,255, 255))
#     cv2.fillConvexPoly(hull_mask, cv2.convexHull(
#                         int_lmrks[68:81]), (255,255, 255))

    nose = proportional_zoom_contour(int_lmrks[27:36], 1.2)
    l_eyes = proportional_zoom_contour(int_lmrks[36:42], 1.8)
    r_eyes = proportional_zoom_contour(int_lmrks[42:48], 1.8)
    mouse = proportional_zoom_contour(int_lmrks[48:60], 1.3)
    
    l_brow = proportional_zoom_contour(int_lmrks[17:22], 1.4)
    r_brow = proportional_zoom_contour(int_lmrks[22:27], 1.4)
    
    
    #face
    for i in range(17-3):
        cv2.fillConvexPoly(
            hull_mask, int_lmrks[i:i+3], (0,0, 0))
        
    # nose
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(nose), (0,0, 0))
    # left eyes
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(l_eyes), (0,0, 0))
    # right eyes
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(r_eyes), (0,0, 0))
    # mouse 
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(mouse), (0,0, 0))
    
    #brow
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(l_brow), (0,0, 0))
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(r_brow), (0,0, 0))
    

    

    return hull_mask

def Recording():
    count = 0
    # open webcam
    vs = WebcamVideoStream().start()
    start = time.time()
    fps = vs.stream.get(cv2.CAP_PROP_FPS)
    print("Frames per second using cv2.CAP_PROP_FPS : {0}".format(fps))

    while True:
        # turn to gray image to detect face
        frame = vs.read()
        img = frame.copy()
        img = imutils.resize(img, width=600)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        if (count % framesSkipping == 0):
            faces = faceDetector(img,0)
        for face in faces:
#             newRectValues = dlib.rectangle(int(face.left() * frame_resize_scale),
#                                int(face.top() * frame_resize_scale),
#                                int(face.right() * frame_resize_scale),
#                                int(face.bottom() * frame_resize_scale))
            predictor = shapePredictor(img, face)
            landmarks = np.matrix([[p.x, p.y] for p in predictor.parts()])
            try:
                faceMask = get_image_hull_mask(img, landmarks)
            except:
                pass
        img = cv2.bitwise_and(faceMask, img)
        cv2.imshow("Frame", img)

        count = count + 1
        # calculate framePerSecond at an interval of 100 frames
        if (count == 100):
            count = 0
        
        # 判斷是否案下"q"；跳離迴圈
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

    #  清除畫面與結束WebCam
    cv2.destroyAllWindows()
    vs.stop()





import re

if __name__ == "__main__":
    
    #Read Image
    FileNameList = []
    for i in (glob.glob(os.path.join("data", "t*"))):
        x = re.search(r'[0-9]+(jpg)?(png)?', i)
        FileNameList.append((x.group(0), i))


    for idx, file in FileNameList:
        Fail = False
    # Read One File
        print("processing " + file)
        Origin_img = cv2.imread(file)
        Origin_img = imutils.resize(Origin_img, width=600)
        img = Origin_img.copy()
        img,Gimg = img2Gray(img)
        
    #     # Method 1 to get face (Skin Detection and Face Detection(Harr))
    #     skinMask = GetSkinMask(img)
    #     faceMask = GetFaceMask(img)
    #     # combine two mask
    #     Face_Skin_Mask = cv2.bitwise_and(skinMask, faceMask)
    #     img = cv2.bitwise_and(Face_Skin_Mask, img)
    #     # can't cut out only face in result

        # Method 2 to get face (Use Dlib)
        SkinMask = GetSkinMask(img)
        try: 
            face, landmarks = get_landmarks(Gimg)
            faceMask = get_image_hull_mask(img, landmarks)
            img = cv2.bitwise_and(faceMask, img)
            # cv2.imwrite("faceMask.jpg", img)
            # showImage(img)
            # img = cv2.bitwise_and(SkinMask, img)
            # not a good method, which cost 1.55 sec


            # img = cv2.imread('./data/acne.jpg')
            ad = Acne_Dector(img)
            ad.run(method =1, debug=False)
            img = cv2.inpaint(Origin_img, ad.mask, 3, cv2.INPAINT_TELEA)
            
        except:
            Fail = True

        if(Fail):
            print("can't process this image")
            print("face should be detectable")
        else:
            saveName = "Results/r" + idx + ".jpg" 
            cv2.imwrite(saveName, img)
    




