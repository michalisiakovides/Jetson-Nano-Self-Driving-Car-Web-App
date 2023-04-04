import cv2
import numpy as np 
from PIL import Image


class camera_1:
  
  def __init__(self):
    self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)

  def callback(self,data):
    bridge = CvBridge()

    try:
      image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    except CvBridgeError as e:
      rospy.logerr(e)
        
    msg = String()
    image = cv2.imread(r"/home/michalis111/Downloads/red.png")
    #cropped image
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Converts images from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])

    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([30,255,255])

    lower_green = np.array([50,100,100])
    upper_green = np.array([70,255,255])

     
     # Here we are defining range of bluecolor in HSV
     # This creates a mask of blue coloured 
     # objects found in the frame.
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
     
     # The bitwise and of the frame and mask is done so 
     # that only the blue coloured objects are highlighted 
     # and stored in res
    red_res = cv2.bitwise_and(image,image, mask= red_mask)
    yellow_res = cv2.bitwise_and(image,image, mask= yellow_mask)
    green_res = cv2.bitwise_and(image,image, mask= green_mask)

     # median blur
    redb= cv2.medianBlur(red_res,3)
    yellowb= cv2.medianBlur(yellow_res,3)
    greenb = cv2.medianBlur(green_res,3)

     #RGB to Grey
    red_grey = cv2.cvtColor(redb,cv2.COLOR_BGR2GRAY)
    yellow_grey = cv2.cvtColor(yellowb,cv2.COLOR_BGR2GRAY)
    green_grey = cv2.cvtColor(greenb,cv2.COLOR_BGR2GRAY)
 

     # hough circle detect
    r_circles = cv2.HoughCircles(red_grey, cv2.HOUGH_GRADIENT,1,20,param1=60,param2=40,minRadius=0,maxRadius=0)
    y_circles = cv2.HoughCircles(yellow_grey, cv2.HOUGH_GRADIENT,1,20,param1=60,param2=40,minRadius=0,maxRadius=0)
    g_circles = cv2.HoughCircles(green_grey, cv2.HOUGH_GRADIENT,1,20,param1=60,param2=40,minRadius=0,maxRadius=0)

    if r_circles is not None:
         #cv2.putText(image,'RED',(0,25), font, 1,(0,0,255),2,cv2.LINE_AA)
         #cv2.imshow('RED',image)
         #cv2.waitKey(0)
         print('Red')
         msg = 'Red'
         pub.publish(msg)
         return msg

    if y_circles is not None:
         #cv2.putText(image,'YELLOW',(0,25), font, 1,(255,255,51),2,cv2.LINE_AA)
         #cv2.imshow('YELLOW',image)
         #cv2.waitKey(0)
         print('Yellow')
         msg = 'Yellow'
         pub.publish(msg)
         return msg

    if g_circles is not None:
         #cv2.putText(image,'GREEN',(0,25), font, 1,(0,255,0),2,cv2.LINE_AA)
         #cv2.imshow('GREEN',image)
         #cv2.waitKey(0)
         print('Green')
         msg = 'Green'
         pub.publish(msg)
         return msg

    else:
         print("No traffic light")
         msg = "No traffic light"
         pub.publish(msg)
         return msg


if __name__ == '__main__':
    rospy.init_node('Light_Detection', anonymous=False)
    pub = rospy.Publisher('light', String, queue_size=10)
    rate = rospy.Rate(5) # 10hz
    camera_1()

    try:
      rospy.spin()
    except KeyboardInterrupt:
      rospy.loginfo("Shutting down")
    
    cv2.destroyAllWindows()

