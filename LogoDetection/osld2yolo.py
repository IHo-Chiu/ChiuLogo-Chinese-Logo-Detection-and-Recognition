
import os
import pickle
from PIL import Image
import shutil

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_paths(root, sub_titles):
    paths = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name[-3:] not in sub_titles :	
                continue
            if '/.i' in path:
                continue
            paths.append(os.path.join(path, name))

    paths.sort()
    return paths

def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    bbox[0] = max(0, bbox[0])
    bbox[1] = max(0, bbox[1])
    bbox[2] = min(w, bbox[2])
    bbox[3] = min(h, bbox[3])
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]
    
src_dir = 'osld'
dst_dir = f'{src_dir}_yolo'
train_image_dir = os.path.join(dst_dir, 'images/train/')
val_image_dir = os.path.join(dst_dir, 'images/val/')
test_image_dir = os.path.join(dst_dir, 'images/test/')
train_label_dir = os.path.join(dst_dir, 'labels/train/')
val_label_dir = os.path.join(dst_dir, 'labels/val/')
test_label_dir = os.path.join(dst_dir, 'labels/test/')

mkdir(dst_dir)
mkdir(os.path.join(dst_dir, 'images'))
mkdir(os.path.join(dst_dir, 'labels'))
mkdir(train_image_dir)
mkdir(val_image_dir)
mkdir(test_image_dir)
mkdir(train_label_dir)
mkdir(val_label_dir)
mkdir(test_label_dir)

train_file = 'annotations/osld-train.pkl'
val_file = 'annotations/osld-val.pkl'
test_file = 'annotations/osld-test.pkl'

train_file = os.path.join(src_dir, train_file)
val_file = os.path.join(src_dir, val_file)
test_file = os.path.join(src_dir, test_file)

with open(train_file, 'rb') as file:
    train_data = pickle.load(file)
with open(val_file, 'rb') as file:
    val_data = pickle.load(file)
with open(test_file, 'rb') as file:
    test_data = pickle.load(file)

image_dir = 'product-images'
image_dir = os.path.join(src_dir, image_dir)

image_paths = get_paths(image_dir, ['jpg'])

image_count = 0
lable_count = 0

for image_path in image_paths:
    image_name = os.path.basename(image_path)

    if 'train' in image_name:
        labels = train_data[image_name]
    elif 'val' in image_name:
        labels = val_data[image_name]
    elif 'test' in image_name:
        labels = test_data[image_name]
    else:
        labels = []
    
    img = Image.open(image_path)
    w, h = img.size[0], img.size[1]

    yolo_labels = []
    for label in labels:
        bbox = xml_to_yolo_bbox(label[0], w, h)
        yolo_label = '0 ' + ' '.join([str(x) for x in bbox])
        yolo_labels.append(yolo_label)

    if 'train' in image_name:
        dst_image_dir = train_image_dir
        dst_label_dir = train_label_dir
    elif 'val' in image_name:
        dst_image_dir = val_image_dir
        dst_label_dir = val_label_dir
    elif 'test' in image_name:
        dst_image_dir = train_image_dir
        dst_label_dir = train_label_dir
        
    shutil.copy(image_path, dst_image_dir)
    label_dst_path = os.path.join(dst_label_dir, image_name[:-3] + 'txt')
    with open(label_dst_path, 'w') as f:
        f.writelines('\n'.join(yolo_labels))

    image_count += 1
    lable_count += len(yolo_labels)
    
print(f'image_count = {image_count}')
print(f'lable_count = {lable_count}')


