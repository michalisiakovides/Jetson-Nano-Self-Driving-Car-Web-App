#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rospy
import tensorflow as tf
from cv_bridge import CvBridge, CvBridgeError
#load the trained model to classify sign
from keras.models import load_model
from PIL import Image
from sensor_msgs.msg import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from std_msgs.msg import String 



model = load_model(r"/home/michalis111/catkin_ws/src/puzzlebotcode/scripts/TSRv2.h5")

#dictionary to label all traffic signs class.
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left',  
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


def grayscale(img):
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  return img
def equalize(img):
  img =cv2.equalizeHist(img)
  return img
def preprocessing(img):
  img = grayscale(img)
  img = equalize(img)
  img = img/255
  return img

class camera_1:
  
  def __init__(self):
    self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)

  def callback(self,data):
    bridge = CvBridge()

    try:
      image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    except CvBridgeError as e:
      rospy.logerr(e)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    threshold = 0.95
    msg = String()

    
        
    sign_image = np.copy(image)
    #cropped_image = region_of_interest(sign_image)
    cropped_image = sign_image[260:410, 520:670] # Slicing to crop the image

    ncropped_image = np.asarray(cropped_image)
    ncropped_image = cv2.resize(ncropped_image, (32, 32))
    ncropped_image = preprocessing(ncropped_image)

    ncropped_image = ncropped_image.reshape(1, 32, 32, 1)
    cv2.putText(sign_image, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(sign_image, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    # PREDICT IMAGE
    predictions = model.predict(ncropped_image)
    classIndex = model.predict(ncropped_image)
    probabilityValue =np.amax(predictions)
    index = np.argmax(predictions)
    if probabilityValue > threshold:    
      #print("Predicted traffic sign is: ", classes[index])
      #cv2.putText(sign_image,str(index)+" "+classes[index], (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
      #cv2.putText(sign_image, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
      #cv2.imshow("Result", sign_image)
      #cv2.waitKey(3)
      msg = classes[index]
      print(msg)
      pub.publish(msg)
      return msg
    else:
      #print("Predicted traffic sign is: No Sign")
      #cv2.putText(sign_image,"No Sign", (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
      #cv2.putText(sign_image, "No Sign", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
      #cv2.imshow("Result", sign_image)
      #cv2.waitKey(3) 
      msg = "No Sign"
      print(msg)
      pub.publish(msg)
      return msg
    
if __name__ == '__main__':
    rospy.init_node('Sign Detection', anonymous=False)
    pub = rospy.Publisher('sign', String, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    camera_1()
  

    try:
      rospy.spin()
    except KeyboardInterrupt:
      rospy.loginfo("Shutting down")
    
    cv2.destroyAllWindows()
  
  

