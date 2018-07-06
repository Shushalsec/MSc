# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 13:06:19 2018

@author: st18l084
"""

import numpy as np
import cv2



# Create a black image, a window and bind the function to window



def choose_a_cell(img):
    ix,iy = -1,-1
# mouse callback function
    def draw_circle(event,x,y,flags,param):
        global ix,iy
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img,(x,y),10,(0,0,255),-1)
            ix,iy = x,y

    atext = 'Double click a cell and press ESC'
    cv2.namedWindow(atext)
    cv2.setMouseCallback(atext, draw_circle)
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(2) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            print(ix,iy)
    cv2.destroyAllWindows()