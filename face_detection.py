import cv2 #for image and video processing
import numpy as np
import os #to handle file related operations
from gtts import gTTS # for text to speech

### Create a function and test it using tester.py#######

def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY) #convert the image to grayscale inorder to avoid any color mismatching
    
    #We use haar cascade because it can be used to superimpose images and find some difference between them
    face_haar_cascade=cv2.CascadeClassifier('E:/python_programs/face detection/haar_Cascade/haarcascade_frontalface_default.xml')
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=2,minNeighbors=5) #returns the rectangular face img iin the gray image
    #scaleFactor detects the image of the particular sized scale,
    #minNeighbours detects the faces nearby

    return faces,gray_img
    #returns rectangle face image and gray image

#it generates labels for the images by the help of haarcascade
def labels_for_training_data(directory):
    faces=[]
    faceID=[]

#training the model
    for path,subdir,filenames in os.walk(directory): #this os.walk goes through the mentioned directory name and labels the images and fetches the fileames and directories in os directory
    
        for filename in filenames :
            if filename.startswith("."): #this is not ot consider the system files
                print("skipping system files")
                continue #doesnt process the system files

            id=os.path.basename(path)#either 0 or 1 in this case gets the basename from the filename or subdirectory names
            img_path=os.path.join(path,filename)#join the path of the image and the filename to img_path
            print("img_path:",img_path)
            print("img_id:",id) #gives the basename
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("image not loaded properly") #sometimes the imread gives null 
                continue
            face_rect,gray_img=faceDetection(test_img)
            if len(face_rect)!=1:
                continue # we will detect only one face
            (x,y,w,h)=face_rect[0] #returns the coordinates for rectangle form haar cascade
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray) #append the gray img to this list
            faceID.append(int(id))#only int will be taken as id
    return faces,faceID
            
#uninstall opencv-python and install opencv-contrib-python for this
def train_classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()#local binary pattern histogram convrts the image to 3X3 matrix and returns binary value depending on algorithm
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,0,255),thickness=3)

def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(0,255,255),2)
    
def to_audio(name,label):
    f="The name of the person(s) is " + name[label]
    language='en'
    audio=gTTS(text=f,lang=language,slow=False)
    audio.save("1.wav")
    os.system("1.wav")
