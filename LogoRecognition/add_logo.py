import os
import numpy as np
from PIL import Image
import argparse
from LogoRecognition import LogoRecognition
from utils import *

import warnings
warnings.filterwarnings("ignore")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add Logo')
    parser.add_argument('image_folder', type=str)
    args = parser.parse_args()

    image_folder = args.image_folder
    
    model = LogoRecognition()
    image_paths = get_paths(image_folder, ['png', 'jpg', 'peg', 'tiff', 'bmp'])

    print(image_paths)
    
    label_names = []
    imgs = []
    for image_path in image_paths:
        label_name = os.path.basename(os.path.dirname(image_path))
        if os.path.exists(image_path):
            print(image_path)
            img = Image.open(image_path)
            img = np.array(img)
            label_names.append(label_name)
            imgs.append(img)

    for label_name, img in zip(label_names, imgs):
        model.add_logo(img, label_name)
    