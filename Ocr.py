import cv2
import numpy as np
import easyocr
import glob
import csv

#setting variables
total_data_pulled = []
existing_part_num = []
photos = glob.glob("Parts/pre_processed/*")


#opening csv file and converting it to a list
with open("part_num.csv", 'r') as file:
    system_part_num = csv.reader(file)
    for row in system_part_num:
        existing_part_num.append(row[0])


# going photo to photo
for photo in photos:

    #setting variables
    data_pulled = []
    img = cv2.imread(photo)
    reader = easyocr.Reader(['en'], gpu=True)

#proccessing image

    #grey scale
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #sharpened image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    clahe_img = clahe.apply(gray_scale)
    clahe_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    clahe_filtered_img = cv2.filter2D(clahe_img, -1, clahe_kernel)

    #inverted image
    otsu_img = cv2.threshold(gray_scale, 0, 255,
	    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    otsu_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    otsu_filtered_img = cv2.filter2D(otsu_img, -1, otsu_kernel)
    

    #adding processed images to list
    processed_images = [clahe_filtered_img, otsu_filtered_img]

    #saving images for viewing
    cv2.imwrite('Parts/post_processed/' + str(photo.split("/")[-1].split(".")[0]) + '_processed.png', cv2.hconcat(([clahe_filtered_img,otsu_filtered_img])))


    # pulling text from proccessed images and exporting the data
    for filtered_img in processed_images:

        results = reader.readtext(filtered_img, rotation_info=[90,180,270], mag_ratio=20)

        for (bbox,text,prob) in results:
            data_pulled.append(text)

        total_data_pulled.append([str(photo.split("/")[-1]),data_pulled])


#setting variables
total_data_formatted = []
total_data_filtered = []
match = []

#formatting ocr data output
for (photo,items) in total_data_pulled:
    for item in items:
        total_data_formatted.append((photo,item))


#clearing duplicates from dataset 
for (photo, value) in total_data_formatted:
    if (photo,value) not in total_data_filtered:
        total_data_filtered.append((photo,value))

#print data
for (photo,value) in total_data_filtered:
    print(str(photo) + " | " +  str(value))

#finding matching parts
for (photo,item) in total_data_filtered:
    characters = item.split(" ")
    for word in characters:
        for existing_part in existing_part_num:
            if word == existing_part:
                if (photo,word) not in match:
                    match.append((photo,word))

#printing results
print("\n\n\n\n\n")
for part in match:
    print(part)
