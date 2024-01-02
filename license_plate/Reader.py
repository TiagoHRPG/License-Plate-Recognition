import cv2
import torch
import easyocr

STRINGLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-"


class Reader:
    def __init__(self) -> None:
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())


    def read_chars(self, rois):
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
    

    def _char_contours_detection(self, img, edged, debug=False):
        drawed_img = img.copy()
        height, width = drawed_img.shape

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = lambda ctr: cv2.boundingRect(ctr)[0])
        drawed_img = cv2.cvtColor(drawed_img, cv2.COLOR_GRAY2BGR)

        filtered_cnts = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            
            # heuristicas para detecção de caracteres
            if height*2  / float(h) > 6: continue
            ratio = h / float(w)
            if ratio < 1.5: continue
            if width / float(w) > 15: continue
        
            cv2.rectangle(drawed_img, (x,y), (x+w, y+h), (255,0,0), 1)
            filtered_cnts.append(c)
        
        if debug:
            cv2.imshow("contours", drawed_img)
            cv2.waitKey(0)
        
        return filtered_cnts

    def read_text(self, plate, debug=False):
        output = self.reader.readtext(plate, allowlist=STRINGLIST, min_size=60)
        plate_text = ""

        area = plate.shape[0] * plate.shape[1]

        for out in output:
            text_bbox, text, _ = out
            text_area = (text_bbox[1][0] - text_bbox[0][0]) *(text_bbox[2][1] - text_bbox[0][1])

            if text_area >= 0.1*area:
                plate_text += f" {text}"
        
        return plate_text

        
    def get_contours(self, plate):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilated_plate = cv2.dilate(plate, kernel, iterations=1)

        edged = cv2.Canny(dilated_plate, 100, 180)
        cnts = self._char_contours_detection(dilated_plate, edged)

        return dilated_plate, cnts


    def read_chars(self, rois):

        output_text = ""
        for roi in rois:
            output = self.reader.readtext(roi, allowlist=STRINGLIST, min_size=60)
            for out in output:
                text_bbox, text, _ = out
                text_area = (text_bbox[1][0] - text_bbox[0][0]) *(text_bbox[2][1] - text_bbox[0][1])
                if text_area >=0.8*len(roi) * len(roi[0]):
                    output_text += text[0]
        return output_text
