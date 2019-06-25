#Write a python Script that captures images from your webcam video stream
#Extracts all faces from the image frame (using haarcascades)
#stores the face info into numpy arrays

# 1. read and show video stream, captures images
# 2. Detect faces and show bounding box
# 3. Flatten the largest face image and save in a numpy array
# 4. repeat the above for multiple people to generate training data
import cv2
import numpy as np 

#initialising the camera
cap = cv2.VideoCapture(0) #0 is the id of the from camera

#loading the haarcase file
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0 #counter
face_data = [] #creating an array
dataset_path = "./face_dataset/" #it is a emppty folder where we will store the frame

file_name = input("Enter the name of person : ")


while True:
	ret,frame = cap.read() #we read what info we are getting from the webcam

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #converts rgb frame into gray frame so that less mem is used

	if ret == False: #if due to any reason frame is not captured do it again
		continue

	#this faces will be a list, each face is a touple
	faces = face_cascade.detectMultiScale(gray_frame,1.3,5) #1.3 is scaling parameter and 5 is the number of neighbors i
	if len(faces) == 0:
		continue

	k = 1

	#out of x,y,w,h we will take w and h of the face to find the area
	#so it will be index x[2]*x[3]
	#we will do reverse sorting so that the largest face comes to the front
	faces = sorted(faces, key = lambda x : x[2]*x[3] , reverse = True) # we are going to do sorting based upon area of the face

	#incriment the skip
	skip += 1

	#we will draw a bounding box
	for face in faces[:1]: 
		x,y,w,h = face

		offset = 5 #taking padding of 5px from each side
		face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_selection = cv2.resize(face_offset,(100,100)) #resize the image into 100*100

		#we will store only the 10th frame
		if skip % 10 == 0:
			face_data.append(face_selection)
			print (len(face_data)) #length of face data, how many faces we have captured


		cv2.imshow(str(k), face_selection)
		k += 1
		
		#we are drawing rect on the frame and then we will show this frame
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) #x+w and y+h are opp coord of x and y 
		#(0,255,0) is the bgr color

	cv2.imshow("faces",frame)

	#this gives a 8 bit int after anding 32bit with 8 bit 
	key_pressed = cv2.waitKey(1) & 0xFF 
	if key_pressed == ord('q'): #ord('q') means ascii value of button q
		break

#convert our face list which is a array into a numpy array
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print (face_data.shape)

#save this data into file system
np.save(dataset_path + file_name, face_data)
print ("Dataset saved at : {}".format(dataset_path + file_name + '.npy'))

cap.release()
cv2.destroyAllWindows()