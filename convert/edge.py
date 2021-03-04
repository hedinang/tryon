import os
import cv2
from PIL import Image
root_dir = '/home/dung/Project/AI/DeepFashion_Try_On/Data_preprocessing/train_color'
sub_dir = '/home/dung/Project/AI/tryon/convert/edge'
folder = os.listdir(root_dir)
for i, f in enumerate(folder):
    img = cv2.imread('{}/{}'.format(root_dir, f), 0)
    ret, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    img = Image.fromarray(img)
    img.save('{}/{}'.format(sub_dir, f))
