import xml.etree.ElementTree as ET
import os
from PIL import Image 
from utils import get_paths, check_and_create_dir

classes = {}
input_dir = "dataset/LogoDet-3K/"
output_dir = "dataset/LogoDet-3K_preprocessed/"

check_and_create_dir(output_dir)

# identify all the xml files in the annotations folder (input directory)
files = get_paths(input_dir, ['xml'])
# loop through each 
total_len = len(files)
for i, fil in enumerate(files):
    
    # parse the content of the xml file
    tree = ET.parse(fil)
    root = tree.getroot()
    
    # check if the label contains the corresponding image file
    image_src_path = fil[:-3] + 'jpg'
    if not os.path.exists(image_src_path):
        print(f"{image_src_path} image does not exist!")
        continue
        
    img = Image.open(image_src_path) 
    print(str(int(i*100/total_len)) + '% ' + image_src_path + ' '*30, end='\r')

    # for every bbox
    for obj in root.findall('object'):
        label = obj.find("name").text
        pil_bbox = [int(x.text) for x in obj.find("bndbox")]

        # if bbox is too small. it might be error case
        if abs(pil_bbox[0]-pil_bbox[2]) < 1 or abs(pil_bbox[1]-pil_bbox[3]) < 1:
            print(abs(pil_bbox[0]-pil_bbox[2]), abs(pil_bbox[1]-pil_bbox[3]), image_src_path)
            continue

        # record class and image count
        if label not in classes:
            classes[label] = 0            
        classes[label] += 1

        # crop
        crop_img = img.crop(pil_bbox)

        # save cropped image 
        image_dst_path = os.path.join(output_dir, f'{label}/{classes[label]}.jpg')
        image_dst_dir = os.path.dirname(image_dst_path)
        check_and_create_dir(image_dst_dir)
        crop_img.save(image_dst_path)

    

print(f'class_size = {len(classes)}' + ' '*40)

crop_img_count = 0
for label in classes.keys():
    crop_img_count += classes[label]

print(f'img_count = {len(files)}')
print(f'crop_img_count = {crop_img_count}')



