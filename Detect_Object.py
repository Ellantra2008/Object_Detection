#import libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('a.jpg')                       # Take image from system
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)        # Convert Coloured image into Grayscale img


template = cv2.imread('temp.jpg', 0)          #assigning template img

height, width = template.shape[::]            # '::'  It accesses every step-th element between indices start (included) and stop (excluded)


res = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR) #'TM_CCORR'=R(x,y)=∑x′,y′(T(x′,y′)⋅I(x+x′,y+y′)) ,for templet matching operation

plt.imshow(res, cmap='gray')


cv2.imshow('a',res)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)    # compairing min and max value and min and max location possibility

top_left = min_loc  
bottom_right = (top_left[0] + width, top_left[1] + height)

cv2.rectangle(img_gray, top_left, bottom_right, (0, 0, 255),2) # assigning color and size for rectangular box


cv2.imshow("Matched image", img_gray)
print('Image Detected')
cv2.waitKey()
cv2.destroyAllWindows()
