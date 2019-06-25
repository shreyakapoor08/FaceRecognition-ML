# Recognise Faces using some classification algorithm - like Logistic, KNN, SVM etx.


# 1. load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y- values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use KNN to find prediction of face (int)
# 5. map the predicted id to name of the user
# 6. Display the predictions on the screen - bounding box and name




import numpy as np
import cv2
import os

########## KNN CODE ############
def distance(v1, v2):
	# Eucledian 
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]
################################

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

dataset_path = "./face_dataset/"

face_data = [] #forms x value of data
labels = [] #forms y value of data
class_id = 0 #labels for the given file
names = {} #dictionary used to create mapping btw id and name


# Dataset prepration
for fx in os.listdir(dataset_path): #iterate over the directory
	if fx.endswith('.npy'):
		names[class_id] = fx[:-4]
		data_item = np.load(dataset_path + fx) #giving file name along with its path, this will load the file
		face_data.append(data_item) #this will be a list

		#create labels for the class
		#suppose we have a file A.npy which will have 10 faces of person A
		#it is basically 10 rows and each rows will have some number of features
		#np.ones is an array of size 10
		# so class_id 0 is multiplied with array of ones so we get zeroes
		#then we increment the class id and again multiply with array of ones and so on the process will go
		#basically for each training point we are creating labels
		target = class_id * np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data, axis=0) #defining axis to define in which direction we want to concatinate
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print(face_labels.shape)
print(face_dataset.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

font = cv2.FONT_HERSHEY_SIMPLEX

#Testing part

while True:
	ret, frame = cap.read() #reading frames
	if ret == False:
		continue
	# Convert frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect multi faces in the image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for face in faces:
		x, y, w, h = face

		# Get the face ROI
		offset = 5
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))

		out = knn(trainset, face_section.flatten())

		# Draw rectangle in the original image
		cv2.putText(frame, names[int(out)],(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

	cv2.imshow("Faces", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()