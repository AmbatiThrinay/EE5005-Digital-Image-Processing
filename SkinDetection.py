# Author : Ambati Thrinay Kumar Reddy

import cv2
import numpy as np
import imutils
import os

class SkinDetector:

    _DEGUB = False
    '''
    implementation of "RGB-H-CbCr Skin Colour Model for Human Face Detection" published by
    Nusirwan Anwar bin Abdul Rahman, Kit Chong Wei and John See
    Faculty of Information Technology, Multimedia University
    johnsee@mmu.edu.my

    Algorithm from: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.718.1964&rep=rep1&type=pdf
    Archived version: https://archive.org/download/10.1.1.718.1964/10.1.1.718.1964.pdf

    '''
    def __init__(self, debug=False) -> None:
        SkinDetector._DEGUB = debug

    def _get_BGR_mask(self,img_BGR):
        '''
        RGB bounding rule (RULE A from the paper)
        
        args :
            BGR_img : image in BGR form

        returns :
            bgr_mask : boolean ndarray mask for the img in BGR form
        '''
        img_B, img_G, img_R  = cv2.split(img_BGR)
        # using reduce() to chain the operation numpy operations
        BRG_Max = np.maximum.reduce([img_B, img_G, img_R])
        BRG_Min = np.minimum.reduce([img_B, img_G, img_R])
        # skin colour illumination's rule for uniform daylight
        mask_1 = np.logical_and.reduce([img_R > 95, img_G > 40, img_B > 20,
                                        BRG_Max - BRG_Min > 15,abs(img_R - img_G) > 15, 
                                        img_R > img_G, img_R > img_B])
        # the skin colour illumination rule under flashlight or daylight lateral
        mask_2 = np.logical_and.reduce([img_R > 220, img_G > 210, img_B > 170,
                                abs(img_R - img_G) <= 15, img_R > img_B, img_G > img_B])
        #mask_1 & mask_2
        BGR_mask = np.logical_or(mask_1, mask_2)

        if SkinDetector._DEGUB :
            BGR_mask = BGR_mask.astype(np.uint8)*255
            img_BGR = cv2.bitwise_and(img_BGR,img_BGR,mask=BGR_mask)
            cv2.imshow("DEBUG::BGR Mask", img_BGR)

        return BGR_mask
    
    def _get_YCrCb_mask(self,img_BGR):
        '''
        YCrCb bounding rule (Rule B from the paper)

        args :
            img_BGR : image in BGR form
        
        returns :
            YCrCb_mask : boolean ndarray mask for the img in YCrCb form
        '''
        img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
        img_Y, img_Cr, img_Cb = cv2.split(img_YCrCb)

        # YCrCb_mask = np.logical_and.reduce([img_Cr <= 1.5862  * img_Cb + 20,
        #                                     img_Cr >= 0.3448  * img_Cb + 76.2069,
        #                                     img_Cr >= -4.5652 * img_Cb + 234.5652,
        #                                     img_Cr <= -1.15   * img_Cb + 301.75,
        #                                     img_Cr <= -2.2857 * img_Cb + 432.85])

        YCrCb_mask = np.logical_and.reduce([img_Cr >= 136, img_Cr <= 173])
        
        if SkinDetector._DEGUB :
            YCrCb_mask = YCrCb_mask.astype(np.uint8)*255
            img_BGR = cv2.bitwise_and(img_BGR,img_BGR,mask=YCrCb_mask)
            cv2.imshow("DEBUG::YCrCb Mask", img_BGR)

        return YCrCb_mask
    
    def _get_HSV_mask(self, img_BGR):
        '''
        HSV bounding rule (Rule C from the paper)

        args :
            img_BGR : image in BGR form
        
        returns :
            HSV_mask : boolean ndarray mask for the img in HSV form
        '''
        img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
        Hue,Sat,Val = cv2.split(img_HSV)
        HSV_mask = np.logical_or(Hue < 50, Hue > 150)

        if SkinDetector._DEGUB :
            HSV_mask = HSV_mask.astype(np.uint8)*255
            img_BGR = cv2.bitwise_and(img_BGR,img_BGR,mask=HSV_mask)
            cv2.imshow("DEBUG::HSV Mask", img_BGR)

        return HSV_mask
    
    def get_mask(self,img_BGR):
        '''
        Skin pixels detection rule (Rule A and Rule B and Rule C)
        
        args :
            BGR_img : image in BGR form

        returns :
            mask : ndarray mask for the img in BGR form in uint8 format
                    0 maps to 0 and 1 maps to 255
        '''

        mask = np.logical_and.reduce([self._get_BGR_mask(img_BGR),
                                      self._get_YCrCb_mask(img_BGR),
                                      self._get_HSV_mask(img_BGR)])
        mask = np.logical_and.reduce([self._get_BGR_mask(img_BGR),
                                      self._get_HSV_mask(img_BGR)])
        mask = self._get_YCrCb_mask(img_BGR)

        mask = mask.astype(np.uint8)*255
        
        if SkinDetector._DEGUB :
            img_BGR = cv2.bitwise_and(img_BGR,img_BGR,mask=mask)
            cv2.imshow("DEBUG::Skin Detection", img_BGR)

        return mask

    def get_face_skin(self,img):

        if img is None : return None
        if not np.any(img) : return None
        
        skin_mask = self.get_mask(img)
        img = cv2.bitwise_and(img,img,mask=skin_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        img = cv2.erode(img,kernel,iterations = 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        img = cv2.dilate(img,kernel,iterations = 1)

        if SkinDetector._DEGUB :
            cv2.imshow("DEBUG::Skin Detection", img)

        return img

def test_with_webcam():
    sking_detector = SkinDetector(debug=True)

    timer = cv2.TickMeter()
    video_capture = cv2.VideoCapture(1)

    frame_queue = [] # size 4

    while True :

        timer.start()

        # Capture frame-by-frame
        _, frame = video_capture.read()

        # Image blending to increase the clarity of low resolution laptop images
        # frame size = 4
        if len(frame_queue) > 3 :
            frame_queue.pop(0)
            frame_queue.append(frame)
            temp_frame = frame_queue[0]
            for i in range(1,len(frame_queue)):
                temp_frame = cv2.addWeighted(frame_queue[i],0.6,temp_frame,0.4,0.0)
            frame = temp_frame
        else : frame_queue.append(frame)

        frame = sking_detector.get_face_skin(frame)

        timer.stop()

        # print Processing time on the screen
        cv2.putText(frame, f"FPS: {timer.getFPS():.2f}",(0,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

        # display the processed frame
        cv2.imshow('Face Detection', frame)

        key_pressed = cv2.waitKey(10)
        if key_pressed == ord('q') : break # press 'q' to exit

    # release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_with_webcam()
    