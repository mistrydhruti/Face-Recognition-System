#Author: Dhruti Mistry 
#Date 
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#to import images
path = 'images'

#to create list of all images
images = []

#to find all names of images
classNames = []

#grabbing list of images in path folder
myList = os.listdir(path)
print(myList)

for cl in myList:
    #reading current image which is basically path 
    # cl is name of image
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)

    #removing .jpg,png,jpeg extensions
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#converting the image into RGB format 
# finding encodings for each image 
def findEncodings(images):
    encodeList = []
    for img in images:
        #converting the image into RGB
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        #find the encodings of each image
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('encoding complete')

#Add entry of each person in csv file 
#adding time and data for each recogized person
def AuthenticateUser(name):
    with open('authenticatedPerson.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

#initializing the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    #reducing the size of image and converting into RGB
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    #find encodings for webcam image 
    #in webcame image we might find multiple images so fo that we will find locations of those faces
    #face_recognition.face_locations(image) - Returns an array of bounding boxes of human faces in a image
    facesCurFrame = face_recognition.face_locations(imgS)

    #sending all locations of faces for encoding
    #Given an image, return the 128-dimension face encoding for each face in the image.
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
    #iterate through all the faces that we have found in current-frame 
    #compare all these faces in current-frame with the encodings that we have found before
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        
        #comparing both images i.e from images and webcam from one of the encodings we have 
        #Compare a list of face encodings against a candidate encoding to see if they match.
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        print(matches)  
        # Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        # for each comparison face. The distance tells you how similar the faces are.
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)

        #getting that image which has least distance between both images
        matchIndex = np.argmin(faceDis)

        #displaying a box and name around current frame faces from webcam
        if matches[matchIndex]:
            name = classNames[matchIndex]
            print(name)
            #find location to create rectangle
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            AuthenticateUser(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)
