import cv2
import easyocr
import numpy as np

points = []
photo = input("whats the name of your photo: ")


def warp_image_bbox(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x,y])

img = cv2.imread(photo)

cv2.imshow('og', img)
cv2.setMouseCallback('og', warp_image_bbox)

cv2.waitKey(0)
cv2.destroyAllWindows()



print(points)

letters = []
numbers = []

reader = easyocr.Reader(['en'], gpu=True)

unwarped_img = cv2.imread(photo)

x_1 = points[0][0]
y_1 = points[0][1]

x_4 = points[1][0]
y_4 = points[1][1]

width, height = 3500 , 1080
pts1 = np.float32([[x_1,y_1], [x_4,y_1], [x_1,y_4], [x_4,y_4]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
warped_img = cv2.warpPerspective(unwarped_img, matrix, (width,height))

img = warped_img

gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
filetered_img = clahe.apply(gray_scale)

cv2.imwrite('filtered.png',filetered_img)


results = reader.readtext(filetered_img,detail=0, rotation_info=[90,180,270], paragraph=True, x_ths=5)


print(results)

print(str(x_1) + ',' + str(y_1))
print("\n")
print(str(x_4) + ',' + str(y_1))
print("\n")
print(str(x_1) + ',' + str(y_4))
print("\n")
print(str(x_4) + ',' + str(y_4))
print("\n")