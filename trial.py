import numpy as np 
import cv2 
import dlib 
import time 
from scipy.spatial import distance as dist 
from imutils import face_utils 



def yawn_function(shape): 
	upper_lip = shape[50:53] 
	upper_lip = np.concatenate((upper_lip, shape[61:64])) 

	lower_lip = shape[56:59] 
	lower_lip = np.concatenate((lower_lip, shape[65:68])) 

	upper_lip_mean = np.mean(upper_lip, axis=0) 
	lower_lip_mean = np.mean(lower_lip, axis=0) 

	dist_bet_lips = dist.euclidean(upper_lip_mean,lower_lip_mean) 
	return dist_bet_lips 

cam = cv2.VideoCapture('') 


#-------Models---------# 
face_detect = dlib.get_frontal_face_detector() 
landmarks_for_identification = dlib.shape_predictor('Model\shape_predictor_68_face_landmarks.dat') 

#--------Variables-------# 
tresh_val_for_yawn = 35
p_time = 0
while True : 
	capture,image_frame = cam.read() 

	if not capture : 
		break


	#---------frames_per_second------------#	 
	capturing_time = time.time() 
	frames_per_second= int(1/(capturing_time-p_time)) 
	p_time = capturing_time 
	cv2.putText(image_frame,f'frames_per_second:{frames_per_second}',(image_frame.shape[1]-120,image_frame.shape[0]-20),cv2.FONT_HERSHEY_PLAIN,2,(0,200,0),3) 

	#------Detecting face------# 
	grey_scale_img = cv2.cvtColor(image_frame,cv2.COLOR_BGR2GRAY) 
	face_detect = face_detect(grey_scale_img) 
	for face in face_detect: 

		shape_of_mat = landmarks_for_identification(grey_scale_img,face) 
		shape = face_utils.shape_to_np(shape_of_mat) 

		upper_lip_marking = shape[48:60] 
		cv2.drawContours(image_frame,[upper_lip_marking],-1,(0, 165, 255),thickness=3) 

		#-------Calculating the lip dist_bet_lips-----# 
		final_dist_lips = yawn_function(shape) 
		# print(final_dist_lips) 
		if final_dist_lips > tresh_val_for_yawn : 
			cv2.putText(image_frame, f'Yawning!',(image_frame.shape[1]//2 - 170 ,image_frame.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,200),2) 


	cv2.imshow('Webcam' , image_frame) 
	if cv2.waitKey(2) & 0xFF == ord('q') : 
		break

cam.release() 

cv2.destroyAllWindows()
