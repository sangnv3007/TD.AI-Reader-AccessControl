from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import os, glob
from pathlib import Path
from ultralytics import YOLO
import cv2
from PIL import Image
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

# Ham load thu vien vietOCR
def vietocr_load():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = './models/transformerocr.pth'
    config['cnn']['pretrained'] = False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch'] = False
    detector = Predictor(config)
    return detector

def ExportYOLOData(folder_name, engine):
    key_name = 'GiayToTuyThan'
    for file in glob.glob(os.path.join(folder_name, '*.jpg')):
        print(f'Processing file: {file}')
        file_name = Path(file).stem
        image = cv2.imread(file)
        results = engine.predict(image,conf=0.65)
        label_dict = {key_name: {}}
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = result.names[box.cls[0].item()]
                if (key_name == class_id):
                    r = box.xyxy[0].astype(int)
                    image_crop = image[r[1]:r[3], r[0]:r[2]]
                    y = (r[1] + r[3]) / 2.0
                    s = detector.predict(Image.fromarray(image_crop))
                    label_dict[class_id].update({s: y})
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
        
        # Đường dẫn tới file .txt bạn muốn lưu dữ liệu vào
        file_path = f"{file}_label.txt"
        # Mở file .txt trong chế độ ghi ('w' là chế độ ghi, 'a' để thêm vào cuối file)
        with open(file_path, 'w') as file:
            text_ocr = label_dict[key_name]
            if (text_ocr != ''):
                file.write(text_ocr)  # Ghi từng dòng dữ liệu vào file,
        print(f'=> Done file: {file}')
        print('----------------------------------------------------------------')
    print('Completed !!!')

if __name__ == "__main__":
    # Load a model
    detector = vietocr_load()
    engine = YOLO(f'./models/MVB23/best.pt')  # pretrained YOLOv8n model
    classes = list(engine.names.values())
    print(type(engine))
    # folder_path = '/home/polaris/ml/DVC/OCR-DVC-ThanhHoa/pdf2img/MVB22/TTHN/'
    # ExportYOLOData(folder_path, engine)




# def draw_results(img_source):
#     for r in results:
#         annotator = Annotator(img_source)
#         boxes = r.boxes
#         for box in boxes:
#             b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
#             c = box.cls
#             annotator.box_label(b, model.names[int(c)])
        
#     img_source = annotator.result()  
#     return img_source

# for result in results:                                         
#     boxes = result.boxes.cpu().numpy()                        
#     for box in boxes: 
#         class_id = result.names[box.cls[0].item()]  
#         print(class_id)                                       
#         r = box.xyxy[0].astype(int)                     
#         crop = img[r[1]:r[3], r[0]:r[2]]
#         cv2.imshow('Cropped', crop)
#         cv2.waitKey(0)

# for result in results:
#     boxes = result.boxes.cpu().numpy()
#     for i, box in enumerate(boxes):
#         r = box.xyxy[0].astype(int)
#         crop = img[r[1]:r[3], r[0]:r[2]]
#         cv2.imwrite(str(i) + ".jpg", crop)

# cv2.imshow('YOLO V8 Detection', im1)
# cv2.waitKey(0)