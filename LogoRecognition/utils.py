import os

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

def check_and_create_dir(dir):
    if not os.path.exists(dir): 
        os.makedirs(dir) 
