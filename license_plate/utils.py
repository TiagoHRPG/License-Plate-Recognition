import cv2, uuid, os, csv
import shutil

def save_results(plates, folder_path):

    if os.path.isdir(folder_path):  
        shutil.rmtree(folder_path)

    os.mkdir(folder_path)

    for plate in plates:
        img_name = f"{uuid.uuid1()}.jpg"
        cv2.imwrite(os.path.join(folder_path, img_name), plate[1])

        csv_filename = os.path.join(folder_path, "plates.csv")
        with open(csv_filename, mode="a", newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([img_name, plate[0]])