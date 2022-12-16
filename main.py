# Face detection and recognition

import cv2
from SkinDetection import SkinDetector
from CropFace import FaceCropper
import os
import numpy as np


def collect_images():
    face_cropper = FaceCropper()
    skin_detector = SkinDetector()

    # create as image directory
    os.mkdir('images')

    print("Collecting images")
    # Initialize Webcam
    video_capture = cv2.VideoCapture(0)

    images_count = 0
    while True and (images_count <= 100):
        # Capture frame-by-frame
        _, frame = video_capture.read()

        frame = face_cropper.get_cropped_face(frame)
        frame = skin_detector.get_face_skin(frame)
        frame = cv2.resize(frame,(200,200))

        # Display the processed frame
        if not (frame is None) : 
            cv2.imshow('Face', frame)
            images_count += 1
            cv2.imwrite(os.path.join('images',f'img{images_count}.png'), frame)
            print(f"Saving as 'image{images_count}.png'")

        key_pressed = cv2.waitKey(10)
        if key_pressed == ord('q') : break # press 'q' to exit
        
        # # press 'space bar' to save image
        # if key_pressed == ord(' ') : pass
    
    # release the capture
    video_capture.release()
    cv2.destroyAllWindows()

model = cv2.face.FisherFaceRecognizer_create()

def train_model():

    Training_data,Labels=[],[]

    data_path1='images'
    img_files1 = [f for f in os.listdir(data_path1)]
    for i, imgfile in enumerate(img_files1):
        image_path = os.path.join(data_path1,imgfile)
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        Training_data.append(np.asarray(image,dtype=np.uint8))
        Labels.append(1)
    
    data_path2 = 'images2'
    img_files2 = [f for f in os.listdir(data_path2)]
    for i, imgfile in enumerate(img_files2):
        image_path = os.path.join(data_path2,imgfile)
        images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        Training_data.append(np.asarray(images,dtype=np.uint8))
        Labels.append(2)

    Labels = np.asarray(Labels,dtype=np.int32)

    model.train(np.asarray(Training_data),np.asarray(Labels))
    print("Model Training Complete")


def prediction():
    
    face_cropper = FaceCropper()
    skin_detector = SkinDetector()

    # Initialize Webcam
    video_capture = cv2.VideoCapture(0)

    while True :
        # Capture frame-by-frame
        _, frame = video_capture.read()

        frame = face_cropper.get_cropped_face(frame)
        frame = skin_detector.get_face_skin(frame)
        frame = cv2.resize(frame,(200,200))

        # Display the processed frame
        if not (frame is None) :
            face = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            result = model.predict(face)
            confidence = int(result[1])
            print(confidence)
            if confidence < 500 :
                cv2.putText(frame, f"Face Recognized",(10,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            else :
                cv2.putText(frame, f"Face Not Recognized",(10,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            
            cv2.imshow('Face', frame)
                

        key_pressed = cv2.waitKey(10)
        if key_pressed == ord('q') : break # press 'q' to exit
        
    
    # release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    collect_images()
    train_model()
    prediction()