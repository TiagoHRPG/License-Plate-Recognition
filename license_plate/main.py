from ultralytics import YOLO
import cv2
import torch

from Reader import Reader
import vision
import utils


def read_plate(plate, mode='all', debug=False):
    if mode == 'all':
        plate_text = reader.read_text(plate, debug)
        
    
    elif mode == 'char':
        drawed_plate, cnts = reader.get_contours(plate)

        rois = vision.get_rois_from_contours(drawed_plate, cnts)

        plate_text = reader.read_chars(rois)
    
    else:
        raise Exception("mode not available, expected mode equals 'char' or 'all'")
    
    return plate_text



def process_plate(plate, debug=False):
    """
    preprocess plate and read image.
    returns the text
    """
    plate = vision.preprocessing(plate)
    if debug:
        cv2.imshow("preprocessing", plate)
        cv2.waitKey(0)

    plate_text = read_plate(plate, 'char', debug)

    return plate_text

# cpu or gpu
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# reader and model instances
reader = Reader()
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
        
        plate_text = process_plate(plate)

        utils.save_results(plate_text.upper(), plate, "assets/plates/plates.csv", "assets/plates")

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

