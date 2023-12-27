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
    
def bb_intersection_over_union(bboxA, bboxB):
    inter_xmin = max(bboxA[0], bboxB[0])
    inter_ymin = max(bboxA[1], bboxB[1])
    inter_xmax = min(bboxA[2], bboxB[2])
    inter_ymax = min(bboxA[3], bboxB[3])

    interArea = max(0, inter_xmax - inter_xmin + 1) * max(0, inter_ymax - inter_ymin + 1)

    bboxA_area = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxB_area = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    iou = interArea / float(bboxA_area + bboxB_area - interArea)

    return iou

def contours_detection(img, edged):
    img_copy = img.copy()
    height, width = img_copy.shape

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = lambda ctr: cv2.boundingRect(ctr)[0])
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)

    filtered_cnts = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        

        if height*2  / float(h) > 6: continue
        ratio = h / float(w)
        if ratio < 1.5: continue
        if width / float(w) > 15: continue
    
        cv2.rectangle(img_copy, (x,y), (x+w, y+h), (255,0,0), 1)
        filtered_cnts.append(c)
    
    return img_copy, filtered_cnts

def get_rois(image, cnts, debug=False):
    # image is binary
    PADDING = 5

    char_rois = []
    char_bboxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        xmin = x-PADDING if x-PADDING >=0 else x
        xmax = x+w+PADDING if x+w+PADDING < image.shape[1] else x+w
        ymin = y-PADDING if y-PADDING >=0 else y
        ymax = y+h+PADDING if y+h+PADDING < image.shape[0] else y+h

        
        duplicate = False
        for i, char in enumerate(char_rois):
            if bb_intersection_over_union(char_bboxes[i], (xmin, ymin, xmax, ymax)) > 0.7:
                duplicate = True
        if duplicate:
            continue

        char_bboxes.append((xmin, ymin, xmax, ymax))
        char = image[int(ymin):int(ymax), int(xmin):int(xmax)]
        char_rois.append(char)
        if debug:
            cv2.imshow("char roi", char)
            cv2.waitKey(0)
    return char_rois

def read_chars(rois):
    global reader, STRINGLIST

    output_text = ""
    for roi in rois:
        output = reader.readtext(roi, allowlist=STRINGLIST, min_size=60)
        for out in output:
            text_bbox, text, _ = out
            text_area = (text_bbox[1][0] - text_bbox[0][0]) *(text_bbox[2][1] - text_bbox[0][1])
            if text_area >=0.8*len(roi) * len(roi[0]):
                output_text += text[0]
    return output_text

def preprocessing(plate):
    scale = 3
    height, width = plate.shape[:2]

    plate = cv2.resize(plate, (width*scale, height*scale))
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, plate_thresh = cv2.threshold(plate_gray, 80, 255, cv2.THRESH_BINARY_INV)

    return plate_thresh

def read_plate(plate, mode='all', debug=False):
    if mode == 'all':
        output = reader.readtext(plate, allowlist=STRINGLIST, min_size=60)
        plate_text = ""

        area = plate.shape[0] * plate.shape[1]

        for out in output:
            text_bbox, text, _ = out
            text_area = (text_bbox[1][0] - text_bbox[0][0]) *(text_bbox[2][1] - text_bbox[0][1])

            if text_area >= 0.1*area:
                plate_text += f" {text}"
        
    
    elif mode == 'char':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilated = cv2.dilate(plate, kernel, iterations=1)

        edged = cv2.Canny(plate, 100, 180)
        dilated_copy, cnts = contours_detection(dilated, edged)

        rois = get_rois(dilated, cnts)

        plate_text = read_chars(rois)
    
    else:
        raise Exception("mode not available, expected mode equals 'char' or 'all'")
    
    return plate_text



def process_plate(plate, debug=False):
    """
    preprocess plate and read image.
    returns the text
    """
    plate = preprocessing(plate)
    if debug:
        cv2.imshow("preprocessing", plate)
        cv2.waitKey(0)

    plate_text = read_plate(plate, 'all', debug)

    return plate_text

# cpu or gpu
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# reader and model instances
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
net = YOLO("assets/best.pt")

# path to images 
img_path = "assets/imgs"


detected_plates = net.predict(img_path, device=DEVICE)
for detected_plate in detected_plates:
    detected_plate = detected_plate.cpu().numpy()

    img = detected_plate.orig_img

    for i, box in enumerate(detected_plate.boxes.xyxy):
        xmin, ymin, xmax, ymax = box

        plate = img[int(ymin):int(ymax), int(xmin):int(xmax),:].copy()

        # draw rect on plate
        img = cv2.rectangle(img, 
                            (int(xmin), int(ymin)),
                            (int(xmax), int(ymax)),
                            (0, 255, 0),
                            2)        
        
        plate_text = process_plate(plate, debug=True)

        save_results(plate_text.upper(), plate, "assets/plates/plates.csv", "assets/plates")

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

