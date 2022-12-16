# Author : Ambati Thrinay Kumar Reddy
# Face detection and cropping the image to contain only face
import cv2
import os
import numpy as np

class FaceCropper:
    
    _DEBUG = False
    def __init__(self, debug=False):
        
        self._frame_queue = []

        # Yunet face classifier
        directory = os.path.dirname(__file__)
        weights = os.path.join(directory, "face_detection_yunet_2022mar.onnx")
        self.face_detector = cv2.FaceDetectorYN_create(weights,"",(0,0))

        FaceCropper._DEBUG = debug
        if FaceCropper._DEBUG :
            cv2.namedWindow("DEBUG::face detection", cv2.WINDOW_AUTOSIZE)

    def _get_cropped_faces(self, img):
        '''
        Function detects the face and returns the cropped face
        
        args :
            img : image on which to find the face
            classifier : harr face classifier
        return :
            list of cropped faces
        '''

        height, width, _ = img.shape
        self.face_detector.setInputSize((width, height))
        _, faces = self.face_detector.detect(img)

        if FaceCropper._DEBUG :
            if faces is not None :
            # drawing the faces on the frame
                img_copy = img.copy()
                for i in range(faces.shape[0]):
                    cv2.rectangle(img_copy, faces[i,:4].astype(np.int16), (0,0,255), 2, cv2.LINE_AA) # face
                    x,y,w,h = faces[i,:4].astype(np.int16)
                    x, y, w, h = x-10, y-10, w+20, h+20
                    cv2.rectangle(img_copy, [x,y,w,h], (0,255,0), 2, cv2.LINE_AA) # face
                    cv2.rectangle(img_copy, faces[i,:4].astype(np.int16), (0,0,255), 2, cv2.LINE_AA) # face
                    cv2.circle(img_copy, faces[i,4:6].astype(np.int16), 2, (0,0,255), 1, cv2.LINE_AA) # left eye
                    cv2.circle(img_copy, faces[i,6:8].astype(np.int16), 2, (0,0,255), 1, cv2.LINE_AA) # right eye
                    cv2.circle(img_copy, faces[i,8:10].astype(np.int16), 2, (0,0,255), 1, cv2.LINE_AA) # nose
                    cv2.circle(img_copy, faces[i,10:12].astype(np.int16), 2, (0,0,255), 1, cv2.LINE_AA) # mouth edge left
                    cv2.circle(img_copy, faces[i,12:14].astype(np.int16), 2, (0,0,255), 1, cv2.LINE_AA) # mouth edge right
                    cv2.imshow('DEBUG::face detection', img_copy)

        if faces is None : return None
        else :
            cropped_faces = []
            faces = faces.astype(np.int16)
            for i in range(faces.shape[0]):
                x,y,w,h = faces[i,:4]
                x,y,w,h = x-10, y-10, w+20, h+20
                cropped_faces.append(img[y:y+h,x:x+w])
            return cropped_faces

    def _blend_image(self,img):
        '''
        Image blending to increase the clarity of low resolution laptop images
        with a img queue of size 3
        '''

        if len(self._frame_queue) >= 3 :
            self._frame_queue.pop(0)
            self._frame_queue.append(img)
            temp_frame = self._frame_queue[0]
            for i in range(1,len(self._frame_queue)):
                temp_frame = cv2.addWeighted(self._frame_queue[i],0.6,temp_frame,0.4,0.0)
            img = temp_frame
        else :
            self._frame_queue.append(img)
        return img
    
    def get_cropped_face(self,img):
        '''
        returns one cropped face image after image blending
        '''

        img = self._blend_image(img)
        faces = self._get_cropped_faces(img)

        if faces is None : return None # no faces are detected
        if len(faces) > 1 : return None # multiples faces are detected
        return faces[0]

def test_with_cam():

    crop_face = FaceCropper(debug=True)

    # Initialize Webcam
    video_capture = cv2.VideoCapture(0)

    while True :
 
        # Capture frame-by-frame
        _, frame = video_capture.read()
        face = crop_face.get_cropped_face(frame)

        # Display the processed frame
        cv2.imshow('Cropped faces', face)

        key_pressed = cv2.waitKey(10)
        if key_pressed == ord('q') : break # press 'q' to exit

    # release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_with_cam()