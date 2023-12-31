import os
import numpy as np
from PIL import Image
import argparse
from LogoRecognition import LogoRecognition
from utils import *

import warnings
warnings.filterwarnings("ignore")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recognition Logo')
    parser.add_argument('image_folder', type=str)
    args = parser.parse_args()

    image_folder = args.image_folder

    model = LogoRecognition()
    image_paths = get_paths(image_folder, ['png', 'jpg', 'peg', 'tiff', 'bmp'])

    for image_path in image_paths:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img = np.array(img)
            
            results = model.run_model(img)

            dimg = model.draw_on(img, results)

            save_dir = 'result'
            check_and_create_dir(save_dir)
            save_name = os.path.join(save_dir, os.path.basename(image_path))
            print(save_name)

            pil_img = Image.fromarray(dimg)
            pil_img.save(save_name)