import datetime
import cv2
import requests
import json
import os
from PIL import Image
import face_recognition
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

print("start")

fast=15
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
process_this_frame=fast
while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    if process_this_frame == fast:
        process_this_frame=0
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        payload={"data" : face_encodings}
        print(payload)
        face_names=[]
        headers = {}
        if len(face_encodings)==0:
            continue
        url ="http://localhost:8000/recognize-faces"
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload,cls=NumpyEncoder))
        face_names=json.loads(response.text)['result']
        print(face_names)
    process_this_frame+=1
    # ind_time = datetime.datetime.now(datetime.timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S.%f')
    ind_time="now"
    # print(face_names,ind_time[11:19])
    print(face_locations,face_names)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
    
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
