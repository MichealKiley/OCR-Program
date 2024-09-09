import cv2
import pytesseract
import numpy as np


img = cv2.imread("motor_label.png")

width, height = 830,315
pts1 = np.float64([[4,51],[409,107],[821,2],[21,256],[822,204]])
pts2 = np.float64([[0,0],[409,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
warped_img = cv2.warpPerspective(img, matrix, (width,height))

cv2.imshow("", warped_img)

cv2.waitKey(0)