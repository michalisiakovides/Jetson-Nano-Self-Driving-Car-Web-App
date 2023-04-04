import cv2
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt


image = cv2.imread(r"/home/michalis111/Downloads/red.png")
#cropped image
font = cv2.FONT_HERSHEY_SIMPLEX
     
# Converts images from BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])

lower_orange = np.array([5,100,100])
upper_orange = np.array([15,255,255])

lower_yellow = np.array([20,100,100])
upper_yellow = np.array([30,255,255])

lower_green = np.array([50,100,100])
upper_green = np.array([70,255,255])

     
     # Here we are defining range of bluecolor in HSV
     # This creates a mask of blue coloured 
     # objects found in the frame.
red_mask = cv2.inRange(hsv, lower_red, upper_red)
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
green_mask = cv2.inRange(hsv, lower_green, upper_green)
     
     # The bitwise and of the frame and mask is done so 
     # that only the blue coloured objects are highlighted 
     # and stored in res
red_res = cv2.bitwise_and(image,image, mask= red_mask)
orange_res = cv2.bitwise_and(image,image, mask= orange_mask)
yellow_res = cv2.bitwise_and(image,image, mask= yellow_mask)
green_res = cv2.bitwise_and(image,image, mask= green_mask)




     # median blur
redb= cv2.medianBlur(red_res,3)
orangeb = cv2.medianBlur(orange_res,3)
yellowb= cv2.medianBlur(yellow_res,3)
greenb = cv2.medianBlur(green_res,3)

#cv2.imshow('RED',redb)
#cv2.waitKey(3)

     #RGB to Grey
red_grey = cv2.cvtColor(redb,cv2.COLOR_BGR2GRAY)
orange_grey = cv2.cvtColor(orangeb,cv2.COLOR_BGR2GRAY)
yellow_grey = cv2.cvtColor(yellowb,cv2.COLOR_BGR2GRAY)
green_grey = cv2.cvtColor(greenb,cv2.COLOR_BGR2GRAY)


     # hough circle detect
r_circles = cv2.HoughCircles(red_grey, cv2.HOUGH_GRADIENT,1,20,param1=60,param2=40,minRadius=0,maxRadius=0)
o_circles = cv2.HoughCircles(orange_grey, cv2.HOUGH_GRADIENT,1,20,param1=60,param2=40,minRadius=0,maxRadius=0)
y_circles = cv2.HoughCircles(yellow_grey, cv2.HOUGH_GRADIENT,1,20,param1=60,param2=40,minRadius=0,maxRadius=0)
g_circles = cv2.HoughCircles(green_grey, cv2.HOUGH_GRADIENT,1,20,param1=60,param2=40,minRadius=0,maxRadius=0)

if r_circles is not None:
     #cv2.putText(image,'RED',(0,25), font, 1,(0,0,255),2,cv2.LINE_AA)
     #cv2.imshow('RED',image)
     #cv2.waitKey(0)
     print('Red')


if o_circles is not None:
     #cv2.putText(image,'ORANGE',(0,25), font, 1,(255,153,51),2,cv2.LINE_AA)
     #cv2.imshow('ORANGE',image)
     #cv2.waitKey(0)
     print('Red')

if y_circles is not None:
     #cv2.putText(image,'YELLOW',(0,25), font, 1,(255,255,51),2,cv2.LINE_AA)
     #cv2.imshow('YELLOW',image)
     #cv2.waitKey(0)
     print('Yellow')

if g_circles is not None:
          #cv2.putText(image,'GREEN',(0,25), font, 1,(0,255,0),2,cv2.LINE_AA)
          #cv2.imshow('GREEN',image)
          #cv2.waitKey(0)
     print('Green')

else:
     print("No traffic light")

"""
fig = plt.figure(figsize=(1, 6))

fig.add_subplot(1, 6, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
fig.add_subplot(1, 6, 2)
plt.imshow(cv2.cvtColor(yellow_mask, cv2.COLOR_BGR2RGB))
plt.title("HSV")
fig.add_subplot(1, 6, 3)
plt.imshow(cv2.cvtColor(yellow_res, cv2.COLOR_BGR2RGB))
plt.title("Masking")
fig.add_subplot(1, 6, 4)
plt.imshow(cv2.cvtColor(yellowb, cv2.COLOR_BGR2RGB))
plt.title("Median Blur")
fig.add_subplot(1, 6, 5)
plt.imshow(cv2.cvtColor(yellow_grey, cv2.COLOR_BGR2RGB))
plt.title("Greyscale")
fig.add_subplot(1, 6, 6)
plt.imshow(cv2.cvtColor(cv2.putText(image,'YELLOW',(0,25), font, 1,(0,255,255),2,cv2.LINE_AA), cv2.COLOR_BGR2RGB))
plt.title("Final")

plt.show(3)
"""


