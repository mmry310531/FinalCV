import cv2
import numpy as np
from sklearn.preprocessing import normalize

class Acne_Dector:
    def __init__(self, img):
        self.img = img

    def show(self, img):
        cv2.imshow('My Image', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, filename='output'):
        cv2.imwrite(f'./Detect_Acne_DB/{filename}.jpg', self.img)

    def run(self):
        img = self.img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # remove influence of mask while normalizing
        #minval = np.min(gray[np.nonzero(gray)]) 
        #gray[np.where(gray == 0)] = minval
        gray[np.where(gray == 0)] = 128
        
        # get norm gray
        normalizedImg = gray
        cv2.normalize(gray,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        self.show(normalizedImg)
        
        # exract brightness
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = img_hsv[:,:,2]
        norm_v = np.zeros_like(v)
        cv2.normalize(v,  norm_v, 0, 255, cv2.NORM_MINMAX)
        self.show(v)
        
        # region of interest
        roi = norm_v - normalizedImg
        self.show(roi)
        
        # binary threshold
        bin_roi = roi.copy()
        threshold = int(0.2 * 255)
        bin_roi[np.where(roi < threshold)] = 0
        bin_roi[np.where(roi >= threshold)] = 255
        
        self.mask = bin_roi
        self.img[np.where(bin_roi!=255)] = 0


if __name__ == '__main__':
    img = cv2.imread('./Detect_Acne_DB/test2.jpg')
    ad = Acne_Dector(img)
    ad.run()
    ad.show(ad.mask)
    ad.show(ad.img)
