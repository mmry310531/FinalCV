import cv2
import numpy as np
from sklearn.preprocessing import normalize

class Acne_Dector:
    def __init__(self, img):
        self.img = img

    def show(self, img, name = 'image'):
        cv2.imshow(name, img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, filename='output'):
        cv2.imwrite(f'./Detect_Acne_DB/{filename}.jpg', self.img)

    def run(self, debug = True):
        img = self.img
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mask = np.where(gray == 0)
        # remove influence of mask while normalizing
        #minval = np.min(gray[np.nonzero(gray)]) 
        #gray[np.where(gray == 0)] = minval
        gray[mask] = 128
        
        # get norm gray
        normalizedImg = gray
        cv2.normalize(gray,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        
        if debug:
            self.show(normalizedImg, 'normalizedImg')
        
        # exract brightness
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = img_hsv[:,:,2]
        norm_v = np.zeros_like(v)
        cv2.normalize(v,  norm_v, 0, 255, cv2.NORM_MINMAX)
        if debug:
            self.show(v, 'v')
        
        # region of interest
        roi = norm_v - normalizedImg
        roi[mask] = 0
        if debug:
            self.show(roi, 'roi')
        
        # binary threshold
        bin_roi = roi.copy()
        threshold = int(0.2 * 255)
        bin_roi[np.where(roi < threshold)] = 0
        bin_roi[np.where(roi >= threshold)] = 255
        
        ret, bin_roi = cv2.threshold(bin_roi, threshold, 255, cv2.THRESH_BINARY)

        if debug:
            self.show(bin_roi, 'bin_roi')
        
        # remove abnormal area
        
        contours, hierarchy = cv2.findContours(bin_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        bin_roi_BGR = cv2.cvtColor(bin_roi, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 70:
                cv2.drawContours(bin_roi_BGR, [cnt], -1, (0, 0, 0), -1)
            
        if debug:
            self.show(bin_roi_BGR, 'bin_roi after cnt')
        
        self.mask = cv2.cvtColor(bin_roi_BGR, cv2.COLOR_BGR2GRAY)
        #self.img[np.where(bin_roi!=255)] = 0
        self.img[np.where(self.mask==255)] = 0



if __name__ == '__main__':
    img = cv2.imread('./Detect_Acne_DB/test2.jpg')
    ad = Acne_Dector(img)
    ad.run(False)
    ad.show(ad.mask, 'mask')
    ad.show(ad.img, 'result')
