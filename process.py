import cv2
import numpy as np
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import time
import os
import re
import glob
import json
from pathlib import Path
from ultralytics import YOLO
from qreader import QReader

# Ham check dinh dang dau vao cua anh
def check_type_image(path):
    imgName = str(path)
    imgName = imgName[imgName.rindex('.')+1:]
    imgName = imgName.lower()
    return imgName

# Ham ve cac boxes len anh
def draw_prediction(img, classes, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes)
    color = (0, 0, 255)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 1)
    cv2.putText(img, label, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Ham load thu vien vietOCR
def vietocr_load():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = './models/transformerocr.pth'
    config['cnn']['pretrained'] = False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch'] = False
    detector = Predictor(config)
    return detector

# Ham load mo hinh Yolo
def load_model_new(yolo_path, ner_path = None):
    # Load a model
    model_yolo = YOLO(yolo_path)  # pretrained YOLOv8n model
    print(f"Load model YOLO from file {yolo_path} susscessfully..")
    model_ner = None
    if ner_path is not None:
        model_ner = spacy.load(ner_path) # Load pretrained NER model
        print(f"Load model NER from file {yolo_path} susscessfully..")
    return model_yolo, model_ner

# Ham xu ly tra ve du liẹu dang dictionary
def process_image(img, engine, label_dict):
    # Thuc hien phat hien vung chua du lieu
    confidence = 0.65 # TODO : Fix cung nguong phat hien doi tuong
    start_time_yolo = time.time()
    results = engine.predict(img,conf=confidence)
    end_time_yolo = time.time()
    print(f'elapsed_time yolo: {end_time_yolo-start_time_yolo}[sec]')
    for result in results:
        boxes = result.boxes.cpu().numpy()
        start_time = time.time()
        for box in boxes:
            class_id = result.names[box.cls[0].item()]
            r = box.xyxy[0].astype(int)
            image_crop = img[r[1]:r[3], r[0]:r[2]]
            y = (r[1] + r[3]) / 2.0
            s = detector.predict(Image.fromarray(image_crop))
            label_dict[class_id].update({s: y})
        end_time = time.time()
        elapsed_time = end_time - start_time
        print ("elapsed_time ocr:{0}".format(elapsed_time) + "[sec]")
    return label_dict

# Ham resize anh 
def resize_image(img, output_image_path=None, new_width=720):

    # Calculate the ratio of the new width to the original width
    ratio = new_width / img.shape[1]

    # Calculate the new height based on the ratio
    new_height = int(img.shape[0] * ratio)

    # Resize the image using the new width and height
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Save the resized image if output path is provided
    if output_image_path:
        cv2.imwrite(output_image_path, resized_img)

    return resized_img

# Ham crop anh theo thu muc
def process_image_folder(engine, folder_path, folder_path_save = None):
    # TODO : Fix cung nguong phat hien doi tuong
    confidence = 0.65 
    # Get classes name from engine
    classes = list(engine.names.values())
    # Path save image crop
    folder_save_crop = folder_path
    if folder_path_save is not None:
        os.makedirs(folder_path_save, exist_ok=True)
        folder_save_crop = folder_path_save
    # Browse each image file
    for filename in os.listdir(folder_path):
        # Check the image format
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            # Full path of image file
            file_path = os.path.join(folder_path, filename)
            # Print processing image
            print("Processing image:", filename)
            img = cv2.imread(file_path)
            results = engine.predict(img,conf=confidence)
            label_boxes = {}
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    # Get label name
                    class_name = result.names[box.cls[0].item()]
                    # Get box coordinates in (left, top, right, bottom) format
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    #draw_prediction(img, class_name, x1, y1, x2, y2)
                    # Add key - labelName, value - center point in bounding boxes
                    label_boxes[class_name] = [(x1+x2)/2, (y1+y2)/2]

            # Check image corner
            if len(label_boxes) < 3:
                print("Error! Unable to find all corners in the image!")
                continue

            # Calculate if corner misses
            crop = img.copy()
            label_misses = find_miss_corner(label_boxes, classes)
            misses_count = len(label_misses)
            if misses_count > 0:
                print(f"WARNING: There are {misses_count} missing corners !")
                label_boxes = calculate_missed_coord_corner(label_misses, label_boxes)
            
            # Crop and save image to folder path
            source_points = np.float32([label_boxes['top_left'], label_boxes['bottom_left'],
                                        label_boxes['bottom_right'], label_boxes['top_right']])
            crop = perspective_transformation(img, source_points)
            
            # Save image crop
            filename_without_extension = os.path.splitext(filename)[0]
            path_save = os.path.join(folder_save_crop, f"{filename_without_extension}_crop.jpg")
            cv2.imwrite(path_save, crop)
        else:
            # If file isn't an image format, skip it
            continue

# Ham ve cac boxes len anh
def draw_prediction(img, classes, x1, y1, x2, y2):
    label = str(classes)
    color = (0, 0, 255)
    # Tạo điểm trung tâm
    center_point = (int((x1+x2)/2), int((y1+y2)/2))
    # Vẽ một hình tròn (điểm trung tâm) với bán kính 5 và màu đỏ (-1 là vẽ đầy đủ)
    cv2.circle(img, center_point, 5, (0, 0, 255), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.putText(img, label, (x1-5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Ham crop image tu 4 goc cua CCCD
async def standardize_id_card(image, classes, confidence):
    # Get predict corner from engine_det
    results = engine_det.predict(image,conf=confidence)
    label_boxes = {}
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            # Get label name
            class_name = result.names[box.cls[0].item()]
            # Get box coordinates in (left, top, right, bottom) format
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            # draw_prediction(img, class_name, x1, y1, x2, y2)
            # Add key - labelName, value - center point in bounding boxes
            label_boxes[class_name] = [(x1+x2)/2, (y1+y2)/2]

    # Check image corner
    if len(label_boxes) < 3:
        # Returns None if more than two labels are missing
        return None

    # Calculate if corner misses
    crop = image.copy()
    label_misses = find_miss_corner(label_boxes, classes)
    if len(label_misses) > 0:
        label_boxes = calculate_missed_coord_corner(label_misses, label_boxes)
    
    # Crop and save image to folder path
    source_points = np.float32([label_boxes['top_left'], label_boxes['bottom_left'],
                                label_boxes['bottom_right'], label_boxes['top_right']])
    crop = perspective_transformation(image, source_points)
    
    # Return image crop
    return crop

# Ham trich xuat thong tin the CCCD
async def info_extraction_VNID(image_path):
    # TODO : Minimum confidence level for detection (HACK)
    conf_predict = 0.65 
    # Get classes name from engine
    classes_det = list(engine_det.names.values())
    classes_rec = list(engine_rec.names.values())
    # Initial return result
    rs = {
        "executionTime": 0,
        "errorCode": None,
        "errorMessage": None,
        "results": []
    }
    # Start time processing an image
    start_time = time.time()
    # Check the image format
    if image_path.endswith(('.jpg', '.png', '.jpeg')):
        # Print processing image
        print("Processing image:", image_path)
        img = cv2.imread(image_path)
        crop = await standardize_id_card(image=img, classes=classes_det, confidence= conf_predict)
        # cv2.imshow('Cropper', crop)
        # cv2.waitKey(0)
        if (crop is not None):
            # Get predict corner from engine_det
            results = engine_rec.predict(crop,conf=conf_predict)
            label_dict = {key: {} for key in classes_rec}
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    class_id = result.names[box.cls[0].item()]
                    r = box.xyxy[0].astype(int)
                    image_crop = crop[r[1]:r[3], r[0]:r[2]]
                    y = (r[1] + r[3]) / 2.0
                    if class_id == 'qr_code':
                        QRs = qreader.detect_and_decode(image=image_crop)
                        print(QRs)
                        continue
                    if class_id == 'mrz':
                        continue
                    s = detector.predict(Image.fromarray(image_crop))
                    label_dict[class_id].update({s: y})
                    
            # Include information in the dictionary
            for key, value in label_dict.items():
                if len(value) >=2: # Gop du lieu
                    sorted_items = sorted(value.items(), key = lambda x:x[1])
                    merged_value = ' '.join([k for k,v in sorted_items])
                    label_dict[key] = merged_value
                elif not value: # Du lieu Null
                    label_dict[key] = None
                else:
                    label_dict[key] = list(value.keys())[0]
            rs = {
                "errorCode": 0,
                "errorMessage": "",
                "results": [label_dict]
            }
        else: 
            # Return error if unable to find all corners in the image
            rs.update({
            "errorCode": 2,
            "errorMessage": "Error! Unable to find all corners in the image!",
            "results": []
            })
    else:
        # If file isn't an image format, skip it
        rs.update({
            "errorCode": 1,
            "errorMessage": "Error! The file is not in the correct format.",
            "results": []
        })
        
    # End time processing an image
    end_time = time.time()
    total_time = end_time - start_time
    # Update execution time 
    rs["executionTime"] = round(total_time,2)

    return rs

# Ham check miss_conner
def find_miss_corner(labels, classes):
    labels_miss = []
    for i in classes:
        bool = i in labels
        if(bool == False):
            labels_miss.append(i)
    return labels_miss

# Ham tinh toan goc miss_conner
def calculate_missed_coord_corner(label_missed, coordinate_dict):
    thresh = 0
    if(label_missed[0]=='top_left'):
        midpoint = np.add(coordinate_dict['top_right'], coordinate_dict['bottom_left']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_right'][0] - thresh
        coordinate_dict['top_left'] = (x, y)
    elif(label_missed[0]=='top_right'):
        midpoint = np.add(coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_left'][0] - thresh
        coordinate_dict['top_right'] = (x, y)
    elif(label_missed[0]=='bottom_left'):
        midpoint = np.add(coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_right'][0] - thresh
        coordinate_dict['bottom_left'] = (x, y)
    elif(label_missed[0]=='bottom_right'):
        midpoint = np.add(coordinate_dict['bottom_left'], coordinate_dict['top_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_left'][0] - thresh
        coordinate_dict['bottom_right'] = (x, y)
    return coordinate_dict

# Ham chuan hoa hinh anh tu cac diem
def perspective_transformation(image, points):
    # Use L2 norm
    width_AD = np.sqrt(
        ((points[0][0] - points[3][0]) ** 2) + ((points[0][1] - points[3][1]) ** 2))
    width_BC = np.sqrt(
        ((points[1][0] - points[2][0]) ** 2) + ((points[1][1] - points[2][1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))  # Get maxWidth
    height_AB = np.sqrt(
        ((points[0][0] - points[1][0]) ** 2) + ((points[0][1] - points[1][1]) ** 2))
    height_CD = np.sqrt(
        ((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))  # Get maxHeight

    output_pts = np.float32([[0, 0],
                             [0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0]])
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(points, output_pts)
    out = cv2.warpPerspective(
        image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    return out

# Ham tra ve thong tin tren mo hinh moi
async def ReturnInfoNew(path, text_code, engine, ner):
    # Tinh thoi gian tai thoi diem bat dau
    start_time = time.time()

    typeimage = check_type_image(path)
    classes = list(engine.names.values())
    label_dict = {key: {} for key in classes}

    # Dau vao la anh co dinh dang(*.jpg,*.jpeg,*.png,...)
    if (typeimage == 'jpg' or typeimage == 'png' or typeimage == 'jpeg'):
        img = cv2.imread(path)
        label_dict.update(process_image(img, engine, label_dict))                                    
    
    # Dau vao la file pdf dang(*.pdf)
    elif(typeimage == 'pdf'):
        image_url = pdf_to_image(path, text_code)
        for url in image_url:
            img = cv2.imread(url)
            label_dict.update(process_image(img, engine, label_dict))
    
    # Dau vao khong dung dinh dang
    else:
        rs = {
            "errorCode": 1,
            "errorMessage": "Lỗi! File không đúng định dạng.",
            "results": []
        }
        return rs
    
    # Gop cac thong tin vao tu dien
    for key, value in label_dict.items():
        if len(value) >=2: # Gop du lieu
            sorted_items = sorted(value.items(), key = lambda x:x[1])
            merged_value = ' '.join([k for k,v in sorted_items])
            label_dict[key] = merged_value
        elif not value: # Du lieu Null
            label_dict[key] = None
        else:
            label_dict[key] = list(value.keys())[0]
    
    # Ham xu ly custom du lieu theo ma van ban 
    label_dict = handle_textcode(label_dict, text_code, ner)

    # Tinh thoi gian tai thoi diem ket thuc thuat toan
    end_time = time.time()
    
    # Tinh tong thoi gian chay
    elapsed_time = end_time - start_time

    # Tra ve ket qua sau khi duyet qua tat ca cac anh
    rs = {
        "errorCode": 0,
        "errorMessage": "",
        "executionTime": round(elapsed_time,2),
        "results": [label_dict]
    }
    return rs

detector = vietocr_load()

qreader = QReader(model_size = 'n',reencode_to="cp65001")
engine_det, ner_det = load_model_new('./models/det/best.pt')
engine_rec, ner_rec = load_model_new('./models/rec/best.pt')

# if __name__ == "__main__":
    #folder_path = '/home/polaris/ml/TD.AI-Reader-AccessControl/images'
    #folder_path_save = '/home/polaris/ml/TD.AI-Reader-AccessControl/images/cropped'
    #process_image_folder(engine, folder_path, folder_path_save)
    # image_path = '/home/polaris/ml/TD.AI-Reader-AccessControl/images/anh_CCCD_Chip (2).jpg'
    # info_extraction_VNID(image_path)