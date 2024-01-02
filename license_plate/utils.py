import cv2, uuid, os, csv

def save_results(text, plate, csv_filename, folder_path):
    img_name = f"{uuid.uuid1()}.jpg"
    cv2.imwrite(os.path.join(folder_path, img_name), plate)

    with open(csv_filename, mode="a", newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text])