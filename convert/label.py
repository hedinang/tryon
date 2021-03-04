from human_parsing import Parsing
import os
import cv2
from PIL import Image
root_dir = '/home/dung/Project/AI/DeepFashion_Try_On/Data_preprocessing/train_img'
sub_dir = '/home/dung/Project/AI/tryon/convert/label'
ref_dir = '/home/dung/Project/AI/tryon/convert/label_ref'
folder = os.listdir(root_dir)
parsing = Parsing()

# palette = [0, 0, 0,
#            15, 15, 15,
#            30, 30, 30,
#            45, 45, 45,
#            60, 60, 60,
#            75, 75, 75,
#            90, 90, 90,
#            105, 105, 105,
#            120, 120, 120,
#            135, 135, 135,
#            150, 150, 150,
#            165, 165, 165,
#            190, 190, 190,
#            205, 205, 205]


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
for i, f in enumerate(folder):
    ref = Image.open('{}/train_label/{}.png'.format(root_dir.split('/train_')[0], f.split('.')[0]))
    img = cv2.imread('{}/{}'.format(root_dir, f), cv2.IMREAD_COLOR)
    img = parsing(img)
    ref.putpalette(palette)
    img.putpalette(palette)
    img.save('{}/{}.png'.format(sub_dir, f.split('.')[0]))
    ref.save('{}/{}.png'.format(ref_dir, f.split('.')[0]))
