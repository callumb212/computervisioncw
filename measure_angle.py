### FANCY FUNCTION

import cv2
import numpy as np
import matplotlib.pyplot as plt

def measure_angle(img_number):
    '''
    Coursework specific function, requires images in working directory saved as image1.png etc.
    
    Will return the calculated angle and number of detected lines.
    
    Will also save visualisation with calculated lines superimposed.
    '''
    
    
    ### Step 1: Skeletonise Image
    # Skeletonisation is probably not essential since top, bottom lines should be parallel anyway
    # but could be a good option to reduce down to two lines
    # Shamelessly stolen: https://medium.com/analytics-vidhya/skeletonization-in-python-using-opencv-b7fa16867331
    
    # Read the image as a grayscale image
    img = cv2.imread(f'image{img_number}.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Threshold the image
    ret, img = cv2.threshold(gray_img, 127, 255, 0)
    
    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    
    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img)==0:
            break
    
    plt.imshow(skel)
    
    
    ### Step 2: Canny & Hough Transform on Skeletonised image
    line_edges = cv2.Canny(skel, threshold1=38.25, threshold2=63.75) # TODO: can we improve edge detection
    lines = cv2.HoughLines(skel,1,np.pi/180,60) # TODO justify choices of threshold etc
    
    ### Step 3: Measure the angle and Create a visualisation of lines on top of original image
    

    for j in range(1, len(lines)):
        for rho,theta in lines[0]: # TODO: clean this up
            theta_i = theta
        for rho,theta in lines[j]:
            theta_j = theta
            
        # TODO: fact check this angle measuring idea
        angle = max(theta_i, theta_j) - min(theta_i,theta_j)
        if angle > 0.2 and angle < np.pi - 0.2: # make sure that it's not picking out 2 lines that are the same line
            break
            
            
    img = cv2.imread(f'image{img_number}.png')
    for final_line in [0, j]:
        for rho,theta in lines[final_line]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
               
           
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    cv2.imwrite(f'houghlines{img_number}.jpg',img)
    plt.imshow(img)
    
    return 360*(angle/(2*np.pi)), lines