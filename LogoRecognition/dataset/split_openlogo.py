import os
from PIL import Image 
import random
from utils import get_paths, check_and_create_dir

random.seed(5201314)

classes = {}
input_dir = "dataset/openlogo_preprocessed/"

# count the number of each logo class
files = get_paths(input_dir, ['jpg'])
total_len = len(files)
for i, fil in enumerate(files):
    label = os.path.basename(os.path.dirname(fil))
    if label not in classes:
        classes[label] = 0
    classes[label] += 1

train_split = []
valid_split = []
test_split = []
remain = []

# if image count < 20, add class to train split
for label in classes.keys():
    if classes[label] < 20:
        train_split.append(label)
    else:
        remain.append(label)

# # train: 64%, valid 16%, test 20%
# test_prob = 0.2
# train_prob = 0.8
# test_size = round(len(classes) * test_prob)
# train_size = round((len(classes) - test_size) * train_prob)
# valid_size = len(classes) - test_size - train_size

# # split
# remain.sort()
# random.shuffle(remain)
# test_split = remain[:test_size]
# valid_split = remain[test_size:test_size+valid_size]
# train_split = remain[test_size+valid_size:] + train_split

# remain is test split
test_split = remain

train_split.sort()
valid_split.sort()
test_split.sort()

# record
print('train split:', len(train_split), len(train_split)/len(classes))
print('valid split:', len(valid_split), len(valid_split)/len(classes))
print('test  split:', len(test_split) , len(test_split)/len(classes))

with open(os.path.join(input_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_split))
with open(os.path.join(input_dir, 'valid.txt'), 'w') as f:
    f.write('\n'.join(valid_split))
with open(os.path.join(input_dir, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_split))


