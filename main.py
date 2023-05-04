import csv
import uuid
import os

from ultralytics import YOLO

import cv2
import numpy as np
import easyocr
import torch

STRINGLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-"
def save_results(text, plate, csv_filename, folder_path):
    img_name = f"{uuid.uuid1()}.jpg"
    cv2.imwrite(os.path.join(folder_path, img_name), plate)

    with open(csv_filename, mode="a", newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text])


def process_plate(plate, bbox_area):
    """
    preprocess plate and read image.
    returns the text
    """
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, plate_thresh = cv2.threshold(plate_gray, 60, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("plate", plate_thresh)
    output = reader.readtext(plate_thresh, allowlist=STRINGLIST, min_size=60)


    if len(output) == 0:
        _, plate_thresh = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)
        cv2.imshow("plate_inv", plate_thresh)
        output = reader.readtext(plate_thresh, allowlist=STRINGLIST, min_size=60)

    plate_text = ""

    for out in output:
        text_bbox, text, _ = out
        text_area = (text_bbox[1][0] - text_bbox[0][0]) *(text_bbox[2][1] - text_bbox[0][1])
        if text_area >= 0.1*bbox_area:
            plate_text += f" {text}"
    
    return plate_text

# cpu or gpu
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# reader and model instances
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
net = YOLO("assets/best.pt")

# path to images 
img_path = "assets/imgs"


results = net.predict(img_path, device=DEVICE, half=True, stream=True)
for result in results:
    result = result.cpu().numpy()

    img = result.orig_img

    for i, box in enumerate(result.boxes.xyxy):
        xmin, ymin, xmax, ymax = box
        bbox_area = (xmax - xmin)*(ymax - ymin)

        plate = img[int(ymin):int(ymax), int(xmin):int(xmax),:].copy()

        # draw rect on plate
        img = cv2.rectangle(img, 
                            (int(xmin), int(ymin)),
                            (int(xmax), int(ymax)),
                            (0, 255, 0),
                            2)        
        
        plate_text = process_plate(plate, bbox_area)

        save_results(plate_text.upper(), plate, "plates.csv", "assets/plates")

        cv2.putText(img, 
                    plate_text,
                    (int(xmin), int(ymin-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2)
        
        cv2.imshow("img", img)
        cv2.waitKey()

del(net)
del(reader)

