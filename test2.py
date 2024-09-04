import cv2
import easyocr
import numpy as np

letters = []
numbers = []

reader = easyocr.Reader(['en'], gpu=True)


unwarped_img = cv2.imread("motor_label.png")

width, height = 1000 , 500
pts1 = np.float32([[25,55],[514,96],[39,265],[500,307]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
warped_img = cv2.warpPerspective(unwarped_img, matrix, (width,height))

img = warped_img

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('greyscale.png', gray)

results = reader.readtext(gray, min_size=20, detail=0)


print(results)




# for (bbox,text,prob) in results:
#     if text.isalpha():
#         letters.append([bbox[0],text])

#     if text.isalpha() == False:
#         numbers.append([bbox[0],text])


# for (bbox,text,prob) in results:
#     print(text)

# for letter,number in letters,numbers:
    



# 5,49
# 242,100
# 492,98
# 717,47

# 25,55
# 514,96
# 39,265
# 500,307