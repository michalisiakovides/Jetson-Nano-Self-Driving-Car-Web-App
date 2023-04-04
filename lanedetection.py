#!/usr/bin/env python

import rospy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

                    
from sensor_msgs.msg import Image
from std_msgs.msg import String 
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32




"""
def compute_steering_angle(frame, lane_lines):
  #Find the steering angle based on lane line coordinate
  #We assume that camera is calibrated to point to dead center
    
    if len(lane_lines) == 0:
      rospy.loginfo('No lane lines detected, do nothing')
      return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        rospy.logdebug('Only detected one lane line, just follow it. %s' % lane_lines[0])
        x1 = lane_lines[0][0]
        x2 = lane_lines[0][2]
        x_offset = x2 - x1
    else:
        left_x2 = lane_lines[0][2]
        right_x2 = lane_lines[1][2]
        camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel
    rospy.logdebug('new steering angle: %s' % steering_angle)
    return steering_angle


def display_heading_line(frame, steering_angle):
  heading_image = np.zeros_like(frame)
  height, width, _ = frame.shape

      # figure out the heading line from steering angle
      # heading line (x1,y1) is always center bottom of the screen
      # (x2, y2) requires a bit of trigonometry

      # Note: the steering angle of:
      # 0-89 degree: turn left
      # 90 degree: going straight
      # 91-180 degree: turn right 
  steering_angle_radian = steering_angle / 180.0 * math.pi
  x1 = int(width / 2)
  y1 = height
  x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
  y2 = int(height / 2)+100

  cv2.line(heading_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
  heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

  return heading_image
"""







def compute_steering_angle(frame, lane_lines):
  msg = String()
  slp = Float32()
  if len(lane_lines) == 0:
      #rospy.loginfo('No lane lines detected, do nothing')
      return "No Lines"

  if len(lane_lines) == 1:
        #rospy.logdebug('Only detected one lane line, just follow it. %s' % lane_lines[0])
        try:
          slope =(lane_lines[0][3]-lane_lines[0][1])/(lane_lines[0][2]-lane_lines[0][0])
          if slope > 0:
            slp = slope
            msg = "Turn Left"
            pub.publish(msg)
            pub1.publish(slp)
            print(slope)
            return msg
          else:
            slp = slope
            msg = "Turn Right"
            pub.publish(msg)
            pub1.publish(slp)
            print(slope)
            return msg
        except:
          #slp = "None"
          msg = "None"
          pub.publish(msg)
          #pub1.publish(slp)
          print(slope)
          return msg
  
  else:
    msg = "Go straight"
    pub.publish(msg)
    return msg
 

def can(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    can = cv2.Canny(blur,455,460)  #455,460 
    return can

def display_lines(image,lines):
  line_image = np.zeros_like(image)
  if lines is not None:
    for line in lines:
      x1,y1,x2,y2 = line.reshape(4)
      cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
      
   
  return line_image

def make_coordinates(image,line_parameters):
  height = image.shape[0]
  width = image.shape[0]
  slope,intercept = line_parameters
  y1 = height  # bottom of the frame
  y2 = int(y1 * (3 / 5))  # make points from middle of the frame down
  x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
  x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
  return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
  height = image.shape[0]
  width = image.shape[0]
  lane_lines=[]
  left_fit = []
  right_fit = []
  boundary = 1/3
  right_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
  left_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

  if lines is None:
        return lane_lines


  for line in lines:
    #x1,y1,x2,y2 = line.reshape(4)
    for x1, y1, x2, y2 in line:
      if x1 == x2:
        continue
      #if y1 == y2:
        #print("stop")
      parameters = np.polyfit((x1,x2),(y1,y2),1)
      #print(parameters)
      slope = parameters[0]
      intercept = parameters[1]
      if slope < 0:
        if x1 < left_region_boundary and x2 < left_region_boundary:
          left_fit.append((slope, intercept))
          #print("Leftslope: ",slope)
         
      else:
        if x1 > right_region_boundary and x2 > right_region_boundary:
          right_fit.append((slope, intercept))
          #print("Rightslope: ",slope)
          
  left_fit_average = np.average(left_fit,axis=0)
  right_fit_average = np.average(right_fit,axis=0)
  if len(left_fit) > 0:
    lane_lines.append(make_coordinates(image, left_fit_average))

  if len(right_fit) > 0:
    lane_lines.append(make_coordinates(image, right_fit_average))

  
  
  #cv2.circle(image, (410,780), radius=5, color=(0, 0, 255), thickness=1)
  return lane_lines


def region_of_interest(image):
  height = image.shape[0]
  triangle = np.array([
  [(2,height-200),(810,height-200),(400,400)]
  ])
  mask = np.zeros_like(image)
  cv2.fillPoly(mask,triangle,255)
  masked_image = cv2.bitwise_and(image,mask)
  return masked_image



class camera_1:
  
  def __init__(self):
    self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)

  def callback(self,data):
    bridge = CvBridge()

    try:
      image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    except CvBridgeError as e:
      rospy.logerr(e)
    
    lane_image = np.copy(image)
    canny = can(lane_image)
    cropped_image = region_of_interest(canny)
    lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,10,np.array([]),minLineLength=30,maxLineGap=10) #minLIneLength=30/ 40,30

    averaged_lines = average_slope_intercept(lane_image,lines)

    line_image = display_lines(lane_image,averaged_lines)

    combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)
    steeringangle = compute_steering_angle(combo_image,averaged_lines)
    cv2.imshow("out",combo_image)
    #cv2.imwrite("/home/michalis111/Pictures/combo4.jpg",combo_image)
    cv2.waitKey(3)
"""
    fig = plt.figure(figsize=(2, 3))

    fig.add_subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(lane_image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    fig.add_subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    plt.title("Greyscale")
    fig.add_subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
    plt.title("Gaussian Blur")
    fig.add_subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(canny, cv2.COLOR_BGR2RGB))
    plt.title("Canny")
    fig.add_subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title("Region of Interest")
    fig.add_subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB))
    plt.title("Hough Lines")
    plt.show(3)
"""
    
      
    
    #print(steeringangle)
    
   


 

if __name__ == '__main__':
    rospy.init_node('Lane_Detection', anonymous=False)
    pub = rospy.Publisher('lane', String, queue_size=10)
    pub1 = rospy.Publisher('slope', Float32, queue_size=10)
    rate = rospy.Rate(5) # 10hz
    camera_1()

    try:
      rospy.spin()
    except KeyboardInterrupt:
      rospy.loginfo("Shutting down")
    
    cv2.destroyAllWindows()
