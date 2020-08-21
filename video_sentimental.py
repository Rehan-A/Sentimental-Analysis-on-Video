#importing libraries
import numpy as np
import cv2
import tensorflow as tf

#used to detect face in your frame  and store the features of face in .xml file 
#for more detail of cv2.CascadeClassifier https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
face_detection = cv2.CascadeClassifier('haar_cascade_face_detection.xml')
profile_face_detection = cv2.CascadeClassifier('haarcascade_profileface.xml')


#caputuring the video mentioned below('gabru.mp4') 
#if you want to run this program for live video put (0) instead of ('gabru.mp4')
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
settings = {
    'scaleFactor': 1.3, 
    'minNeighbors': 5, 
    'minSize': (50, 50)
}


#classifying into 5 classes
labels = ["Neutral","Happy","Sad","Surprise","Angry"]


#loading the expression model 
model = tf.keras.models.load_model('expression.model')


#Refer for more details https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
while True:
	ret, img = camera.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	detected = face_detection.detectMultiScale(gray, **settings)
    
	for x, y, w, h in detected:
		cv2.rectangle(img, (x, y), (x+w, y+h), (245, 135, 66), 2)
		cv2.rectangle(img, (x, y), (x+w//3, y+20), (245, 135, 66), -1)
		face = gray[y+5:y+h-5, x+20:x+w-20]
		face = cv2.resize(face, (48,48)) 
		face = face/255.0
		
		predictions = model.predict(np.array([face.reshape((48,48,1))])).argmax()
		state = labels[predictions]
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,state,(x+10,y+15), font, 0.5, (255,255,255), 2, cv2.LINE_AA)

	# profile face detection
	detected = profile_face_detection.detectMultiScale(gray, **settings)
	for x, y, w, h in detected:
		cv2.rectangle(img, (x, y), (x+w, y+h), (245, 135, 66), 2)
		cv2.rectangle(img, (x, y), (x+w//3, y+20), (245, 135, 66), -1)
		face = gray[y+5:y+h-5, x+20:x+w-20]
		face = cv2.resize(face, (48,48)) 
		face = face/255.0
		
		predictions = model.predict(np.array([face.reshape((48,48,1))])).argmax()
		state = labels[predictions]
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,state,(x+10,y+15), font, 0.5, (255,255,255), 2, cv2.LINE_AA)

	img = cv2.resize(img,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_AREA)
	cv2.imshow('Facial Expression', img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	writer = cv2.VideoWriter()
	writer.write(img)
	if cv2.waitKey(5) != -1:
		break

camera.release()
cv2.destroyAllWindows()
