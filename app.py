import streamlit as st
import cv2
from SkinDetection import SkinDetector
from CropFace import FaceCropper
import os
import numpy as np

model = cv2.face.FisherFaceRecognizer_create()

def train_model():

    Training_data,Labels=[],[]

    data_path1='images1'
    img_files1 = [f for f in os.listdir(data_path1)]
    for i, imgfile in enumerate(img_files1):
        image_path = os.path.join(data_path1,imgfile)
        images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        Training_data.append(np.asarray(images,dtype=np.uint8))
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
    st.session_state.model = model

    print("Model Training Complete")

def img_messages(img_size,message):
    img = np.zeros(img_size, dtype = np.uint8)
    cv2.putText(img, message,(10,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    return img

train_model()

## App for GUI
st.set_page_config(layout="wide", page_title="Face Recognition for Authentication")
st.markdown('''
<div align="center">

## **EE5005 Digital Image Processing**

### **Face Recognition for Authentication**
**By**

**Ambati Thrinay Kumar Reddy - 121901003**

**Peddakotla Rohith - 121901036**

---
</div>''',unsafe_allow_html=True)


st.write("Stay still for collecting the images for face registration")


col1, col2, col3 = st.columns(3)
loading_msg = img_messages((200,300,3),"Loading, Please wait ...")

col1.write('### Webcam Live Feed')
WEBCAM_WINDOW = col1.image([cv2.cvtColor(loading_msg, cv2.COLOR_BGR2RGB)])
col2.write('### Face DETECTION and cropping')
CROPPED_WINDOW = col2.image([cv2.cvtColor(loading_msg, cv2.COLOR_BGR2RGB)])
col3.write('### Skin Detection')
SKIN_DETECTION_WINDOW = col3.image([cv2.cvtColor(loading_msg, cv2.COLOR_BGR2RGB)])

camera = cv2.VideoCapture(0)

face_cropper = FaceCropper()
skin_detector = SkinDetector()

while True:
    try :
        _, frame = camera.read()

        cropped_img = face_cropper.get_cropped_face(frame)

        # No face was detected
        if cropped_img is None :
            no_face_msg = img_messages((300,500,3),'NO Face Detected')
            CROPPED_WINDOW.image(cv2.cvtColor(no_face_msg, cv2.COLOR_BGR2RGB))
            SKIN_DETECTION_WINDOW.image(cv2.cvtColor(no_face_msg, cv2.COLOR_BGR2RGB))
            continue
        else :
            CROPPED_WINDOW.image(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))


        skin_img = skin_detector.get_face_skin(cropped_img)
        SKIN_DETECTION_WINDOW.image(cv2.cvtColor(skin_img, cv2.COLOR_BGR2RGB))

        face = cv2.cvtColor(skin_img.copy(), cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face,(200,200))
        result = model.predict(face)
        confidence = int(result[1])
        print(confidence)
        
        WEBCAM_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    except :
        # Incorrect Image format or size
        pass




    
    