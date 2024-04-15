from qreader import QReader
import cv2
import time

qreader = QReader(model_size = 'n',reencode_to="cp65001")

start_time = time.time()
# Read the image
image1 = cv2.imread('/home/polaris/ml/TD.AI-Reader-AccessControl/images/anh_CCCD_Chip (2).jpg')
#img = cv2.cvtColor(cv2.imread('/content/sample_data/anh_CCCD_Chip (201)_crop.jpg'), cv2.COLOR_BGR2RGB)
# Detect and decode the QRs within the image
QRs = qreader.detect_and_decode(image=image1)
end_time = time.time()
print(f'{end_time-start_time} [sec]')
# Print the results
for QR in QRs:
    print(QR)