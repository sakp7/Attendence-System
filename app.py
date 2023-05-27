import cv2
import streamlit as st
import os
import numpy as np
from datetime import datetime
import face_recognition
from streamlit_option_menu import option_menu
import pandas as pd
# Function to find face encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Function to mark attendance in Excel
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def main():
    st.set_page_config(page_title='Attendance System', layout='wide')
    st.sidebar.header("Welcome")
    st.title("Attendance Portal")

    a1=st.empty()
    a2=st.empty()
    a3=st.empty()
    a4=st.empty()
    a5=st.empty()
    a6=st.empty()
    a7=st.empty()

    with st.sidebar:
        selec=option_menu(
            menu_title="Welcome",
            options=["Home","Register","Login","View Attendence"]
            )
    
    if selec == "Register":



        a1.header("Register")
        name = a2.text_input("Enter Your Name:")
        roll = a3.text_input("Enter Your Roll Number:")
        rsub = a4.button("SUBMIT")
        
        if rsub:
            a5.subheader("Get ready for a photo sample!")
            cam = cv2.VideoCapture(0)
            cam.set(3, 640)
            cam.set(4, 480)
            face_detector = cv2.CascadeClassifier('ha2.xml')
            count = 0
            
            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1
                    cv2.imwrite(f"dataset/{name}" + ".jpg", gray[y:y + h, x:x + w])
                    cv2.imshow('image', img)
                
                k = cv2.waitKey(100) & 0xff
                if k == 27:
                    break
                elif count >= 1:
                    a7.image(img, channels="BGR")
                    break
            
            cam.release()
            a6.write("Registration is successful")
    
    elif selec == "Login":
        st.header("Login")
        a1.empty()
        a2.empty()
        a3.empty()
        a4.empty()
        a5.empty()
        a6.empty()
        a7.empty()
        runcam()
    elif selec=="View Attendence":
        df=pd.read_csv("Attendance.csv")
        st.dataframe(df)



def runcam():
    st.title("Live Webcam Stream")
    
    # Load training images and encodings
    path = 'Training_images'
    images = []
    classNames = []
    myList = os.listdir(path)
    
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')
    
    cap = cv2.VideoCapture(0)
    
    # Create a placeholder for the video stream
    video_placeholder = st.empty()
    
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
    
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)
    
        # Display the frame in Streamlit
        video_placeholder.image(img, channels="BGR")
    
        if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
            break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

main()
