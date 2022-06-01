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
        cv2.imwrite(f'./Detect_Acne_DB/{filename}_mask.jpg', self.mask)
        cv2.imwrite(f'./Detect_Acne_DB/{filename}.jpg', self.img)

    def run(self, method = 1, debug = True):
        assert method in range(2)
        
        img = self.img
        
        if method == 0:
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
            
            '''
            if debug:
                for i in range(5):
                    bin_roi = roi.copy()
                    threshold = int((0.1 + 0.02*i) * 255)
                    bin_roi[np.where(roi >= threshold)] = 100 + 20*i
                    self.show(bin_roi, 'bin_roi threshold')
            '''
            
            threshold = int(0.13 * 255)
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
                if debug and area > 0:
                    print(area)
                if area > 500:
                    cv2.drawContours(bin_roi_BGR, [cnt], -1, (0, 0, 0), -1)
                
            if debug:
                self.show(bin_roi_BGR, 'bin_roi after cnt')
            
            self.mask = cv2.cvtColor(bin_roi_BGR, cv2.COLOR_BGR2GRAY)
            #self.img[np.where(bin_roi!=255)] = 0
            if debug:
                self.img[np.where(self.mask==255)] = (0,0,255)
            else:
                self.img[np.where(self.mask==255)] = 0  
                
        elif method == 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = np.where(gray == 0)
            
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)            
            A = img_lab[:, :, 1]
            
            kernel_size = 19            
            kernel = np.ones((kernel_size,kernel_size), np.float32)/500
            
            # remove the effect of the black area while doing the low-pass filter
            A_rm_mask = A.copy()
            A_rm_mask[mask] = 140
            low_pass = cv2.filter2D(A_rm_mask ,-1, kernel)
            
            diff = A-low_pass
            
            threshold = 42
            ret, bin_roi = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            # remove abnormal area
            contours, hierarchy = cv2.findContours(bin_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            bin_roi_BGR = cv2.cvtColor(bin_roi, cv2.COLOR_GRAY2BGR)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if debug and area > 0:
                    print(area)
                if area > 300:
                    cv2.drawContours(bin_roi_BGR, [cnt], -1, (0, 0, 0), -1)
                elif len(cnt) > 0:
                    try:
                        cnt_2d = np.reshape(cnt, (cnt.shape[0], 2))
                        cnt_2d = proportional_zoom_contour(cnt_2d, 1.2 if area > 100 else 1.5)
                        if cnt_2d.size > 0:
                            cv2.drawContours(bin_roi_BGR, [cnt_2d], -1, (255, 255, 255), -1)
                    except ValueError:
                        pass
                    
            self.mask = cv2.cvtColor(bin_roi_BGR, cv2.COLOR_BGR2GRAY)
            
            if debug:
                self.show(A, 'A')
                self.show(low_pass, 'low_pass')
                self.show(diff, 'diff')
                self.show(bin_roi, 'bin')

            
            self.img[np.where(self.mask==255)] = 0  
            self.img = cv2.inpaint(img, self.mask, 3, cv2.INPAINT_TELEA)
            
            self.mask = bin_roi_BGR

import pyclipper

def perimeter(poly):
    p = 0
    nums = poly.shape[0]
    for i in range(nums):
        p += abs(np.linalg.norm(poly[i % nums] - poly[(i + 1) % nums]))
    return p

def proportional_zoom_contour(contour, ratio):
    """
    多边形轮廓点按照比例进行缩放
    :param contour: 一个图形的轮廓格式[[[x1, x2]],...],shape是(-1, 1, 2)
    :param ratio: 缩放的比例，如果大于1是放大小于1是缩小
    :return:
    """
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

if __name__ == '__main__':
    img = cv2.imread('./data/acne.jpg')
    ad = Acne_Dector(img)
    ad.run(method =1, debug=False)
    ad.show(ad.mask, 'mask')
    ad.show(ad.img, 'result')
    ad.save()
