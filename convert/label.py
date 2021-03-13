from human_parsing import Parsing
import os
import cv2
from PIL import Image
import json
from tqdm import tqdm
from estimator import BodyPoseEstimator
root_dir = '/home/dung/Project/AI/DeepFashion_Try_On/Data_preprocessing/ACGPN_traindata/train_img'
label_dir = '/home/dung/Project/AI/tryon/convert/train_label'
pose_dir = '/home/dung/Project/AI/tryon/convert/train_pose'
edge_dir = '/home/dung/Project/AI/tryon/convert/train_edge'
folder = os.listdir(root_dir)
parsing = Parsing()
# f = '013312_0.jpg'
# image_src = cv2.imread('{}/{}'.format(root_dir, f), cv2.IMREAD_COLOR)
# img = parsing(image_src)
# img.save('{}/{}.png'.format(label_dir, f.split('.')[0]))
# estimator = BodyPoseEstimator(
#     '/home/dung/Project/AI/tryon/checkpoints/openpose_body_coco_pose_iter_440000.pth')
# pose_data = estimator(image_src)
# with open('{}/{}_keypoints.json'.format(pose_dir, f.split('.')[0]), 'w') as outfile:
#     json.dump(pose_data.tolist(), outfile)
# print('aaa')

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


palette = get_palette(20)

# folder = ['013624_0.jpg']
for i, f in enumerate(tqdm(folder)):
    image_src = cv2.imread('{}/{}'.format(root_dir, f), cv2.IMREAD_COLOR)
    # human parsing
    # img = parsing(image_src)
    # img.save('{}/{}.png'.format(label_dir, f.split('.')[0]))

    # pose
    estimator = BodyPoseEstimator(
        '/home/dung/Project/AI/tryon/checkpoints/openpose_body_coco_pose_iter_440000.pth')
    pose_data = estimator(image_src)
    with open('{}/{}_keypoints.json'.format(pose_dir, f.split('.')[0]), 'w') as outfile:
        json.dump(pose_data.tolist(), outfile)
    #edge
    # image_src = cv2.imread('{}/{}'.format(root_dir, f), 0)
    # ret, image_src = cv2.threshold(image_src, 240, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite('{}/{}'.format(edge_dir, f), image_src)
