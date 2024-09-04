import cv2
import easyocr
import numpy as np
import glob
import csv

total_data_pulled = []

photos = glob.glob("Parts/*")


for photo in photos:
    data_pulled = []

    img = cv2.imread(photo)

    reader = easyocr.Reader(['en'], gpu=True)

    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    filetered_img = clahe.apply(gray_scale)

    cv2.imwrite('filtered.png',filetered_img)

    results = reader.readtext(filetered_img, rotation_info=[90,180,270], mag_ratio=20)

    for (bbox,text,prob) in results:
        if prob >= .7:
            data_pulled.append(text)

    total_data_pulled.append([str(photo),data_pulled])


existing_part_num = []
match = []

with open("motor_test1.csv", 'r') as file:
    system_part_num = csv.reader(file)
    for row in system_part_num:
        existing_part_num.append(row[0])

for part_num,ocr_data in total_data_pulled:
    for word in ocr_data:
        print(word)

        for num in existing_part_num:
            if word == num:
                match.append([part_num,word])


for part in match:
    print(part)
