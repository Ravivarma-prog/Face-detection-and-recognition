import cv2
import os,time
import numpy as np
import face_detection as fr
# from gtts import gTTs

#lets load the image
test_img=cv2.imread('E:/python_programs/face detection/test/Image.jpg') #loading the image
faces_detected,gray_img=fr.faceDetection(test_img) #returns the rectangular face and the gray image

faces,faceID=fr.labels_for_training_data('E:/python_programs/face detection/training')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save('trainingData.yml')
name={0:"person1",1:"person2",2:"person3",3:"person4"}
# face_recognizer=cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read('E:/python_programs/face detection/trainingData.yml')

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+w,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    
    if (confidence>100):
        continue
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)
    # #resize the image inorder to fit the rectangle
    resized_img=cv2.resize(test_img,(1000,700)) # resizes te image to 1000X700
    cv2.imshow("face_detected:",resized_img)
    time.sleep(1)
    fr.to_audio(name,label)
    f=open('name.txt','w')
    f.write("the person/s is {}".format(name[label]))
    f.close()




cv2.waitKey()
# cv2.release()
cv2.destroyAllWindows()

