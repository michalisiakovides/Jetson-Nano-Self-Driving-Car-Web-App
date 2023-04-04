#!/usr/bin/env python

import rospy
import cv2
                  
from sensor_msgs.msg import Image
from std_msgs.msg import String 

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
#from lightdetection import lightdet
#from signrecognition import signrec
#from lanedetection import lanedetection
#from lanedetection import average_slope_intercept

"""
class camera_1:
  
  def __init__(self):
    self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)


  def callback(self,data):
    bridge = CvBridge()

    try:
      image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8") 
    except CvBridgeError as e:
      rospy.logerr(e)

    #print(lightdet(image))
    #print(signrec(image))
    lanedetection(image)
    #cv2.imshow("Cropped",lanedetection(image))
    #cv2.waitKey(3)
    
    msg = Twist()
    
    if signrec(image)== "Ahead only":
      msg.linear.x = 0.0
      pub.publish(msg)
      rospy.sleep(5)
      msg.linear.x = 0.2
      #pub.publish(msg)
    
    if lanedetection(image) == "Go straight":
      print("Go straight")
      msg.linear.x = 0.1
      #pub.publish(msg)

    elif lanedetection(image) == "Turn Left":
      print("Turn Left")
      msg.linear.x = 0.1
      msg.angular.z = 0.4
      #pub.publish(msg)
    elif lanedetection(image) == "Turn Right":
      print("Turn Right")
      msg.linear.x = 0.1
      msg.angular.z = -0.4
      #pub.publish(msg)
    else:
      print("No Lines")
      msg.angular.z = 0.0
      msg.linear.x = 0.0
    
    pub.publish(msg)
    
   
    
    
    #pub.publish(msg)
"""



lane = None
slope = None
sign = None
right = None
left = None

def wlcallback(smsgl:Float32):
    global left
    left = smsgl.data
    
def wrcallback(smsgr:Float32):
    global right
    right = smsgr.data

def callbacklane(data):
    global lane
    lane = data.data

def callbackslope(data):
    global slope
    slope = data.data
   
def callbacksign(data):
    global sign
    sign = data.data

def callbacklight(data):    
   global light
   light = data.data   

def main():

  try:
    
    rospy.spin()
  except KeyboardInterrupt:
    rospy.loginfo("Shutting down")
  
  cv2.destroyAllWindows()



if __name__ == '__main__':
    rospy.init_node('Navigation', anonymous=False)
    rate = rospy.Rate(10) # 10hz
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    sublane = rospy.Subscriber("/lane",String,callback=callbacklane)
    subslope = rospy.Subscriber("/slope",Float32,callback=callbackslope) 
    subsing = rospy.Subscriber("/sign",String,callback=callbacksign)
    subwl=rospy.Subscriber("/wl",Float32,callback=wlcallback)
    subwr=rospy.Subscriber("/wr",Float32,callback=wrcallback)
    sublight = rospy.Subscriber("/light",String,callback=callbacklight)
    msg = Twist()
    dt = 0.1
    total = 0.0
    totalth=0.0
    global speed
    speed = 0.1
   


while not rospy.is_shutdown():
    
    if sign == 'End speed + passing limits':
       speed = 0.15

    if lane == "Go straight" or sign == "Ahead only":    #and light=="Green"
        msg.linear.x = speed
      
        #print("Go straight")
        #print(slope)
        
        #pub.publish(msg)

    elif lane == "Turn Left" or lane == "Turn Right":
        #print("Turn Left")
        #print(slope)
        msg.linear.x = speed
        msg.angular.z = 0.25*(1/slope)
        print(msg.angular.z)
        #pub.publish(msg)

    else:                               #for red light
        #print("No Lines")
        #print(slope)
        msg.angular.z = 0.0
        msg.linear.x = 0.0
        #pub.publish(msg)
    
    if sign == "Stop":
      msg.linear.x = 0.0
      msg.angular.z = 0.0
      pub.publish(msg)
      rospy.sleep(5)
      continue
    
    if sign == "Turn right ahead":
      msg.linear.x = 0.0
      msg.angular.z = 0.0
      pub.publish(msg)
      rospy.sleep(2)
      msg.linear.x = 0.15
      pub.publish(msg)
      rospy.sleep(1)
      msg.angular.z = -0.5
      msg.linear.x = 0.0
      pub.publish(msg)
      rospy.sleep(3)
      continue
      
    #print(lane)
    #rospy.loginfo(lane)
    #rospy.loginfo("Speed ",speed)
    pub.publish(msg)
    rate.sleep()
  