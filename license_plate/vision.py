import cv2

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

def get_rois_from_contours(image, cnts, debug=False):
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

def preprocessing(plate, scale=3):
    height, width = plate.shape[:2]

    plate = cv2.resize(plate, (width*scale, height*scale))
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, plate_thresh = cv2.threshold(plate_gray, 80, 255, cv2.THRESH_BINARY_INV)

    return plate_thresh