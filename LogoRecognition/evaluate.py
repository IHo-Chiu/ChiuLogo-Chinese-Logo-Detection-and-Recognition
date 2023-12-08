import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from LogoDataset import LogoDataset
from torch.utils.data import DataLoader
import ChineseCLIP.cn_clip.clip as clip
from ChineseCLIP.cn_clip.clip import load_from_name, available_models

from STR.strhub.data.module import SceneTextDataModule
from STR.strhub.models.utils import load_from_checkpoint, parse_model_args
from ChineseCLIP.cn_clip.clip.model import convert_models_to_fp32

from LogoModel import LogoModel
import torch.nn.functional as F
    
if __name__ == "__main__":
    model_path = "best.ckpt"
    dataset_dir = 'dataset/LogoDet-3K_preprocessed'
    # dataset_dir = 'dataset/openlogo_preprocessed'
    dataset_split = 'test'
    gallery_dataset_type = 'gallery'
    query_dataset_type = 'query'
    # gallery_dataset_type = 'all'
    # query_dataset_type = 'all'
    batch_size=128
    num_workers=8
    beta = 70

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LogoModel.load_from_checkpoint(model_path).to(device)
    model.eval()
    
    gallery_dataset = LogoDataset(dir=dataset_dir, split=dataset_split, type=gallery_dataset_type, transform=model.preprocess)
    gallery_dataloader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                                  collate_fn=gallery_dataset.collate_fn, pin_memory=True, drop_last=False)

    query_dataset = LogoDataset(dir=dataset_dir, split=dataset_split, type=query_dataset_type, transform=model.preprocess)
    query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                                    collate_fn=query_dataset.collate_fn, pin_memory=True, drop_last=False)

    print(f'{gallery_dataset}: {len(gallery_dataset)}')
    print(f'{query_dataset}: {len(query_dataset)}')

    gallery_features = []
    gallery_labels = []
    
    with torch.no_grad():
        for data in tqdm(gallery_dataloader):
            labels, images, image_paths = data
            
            labels = labels.to(device)
            images = images.to(device)

            features = model(images)
            
            gallery_features.append(features)
            gallery_labels.append(labels.unsqueeze(1))

    gallery_features = torch.vstack(gallery_features).cuda()
    gallery_labels = torch.vstack(gallery_labels).squeeze().float().cuda()
    
    correct_total = 0
    text_total = 0
    text_error_total = 0

    with torch.no_grad():
        for data in tqdm(query_dataloader):
            labels, images, image_paths = data
            
            images = images.to(device)
            features = model(images)

            predict_labels = model.predict(features, gallery_features, gallery_labels, beta)
            labels = np.array(labels)
            
            correct_map = (labels == predict_labels.cpu().numpy())
            correct = np.sum(correct_map)
            correct_total += correct

            for idx, correct in enumerate(correct_map):
                query_image = image_paths[idx]
                query_label = os.path.basename(os.path.dirname(query_image))

                if '_text' in query_label:
                    text_total += 1
                    if correct == False:
                        text_error_total += 1
            
    accuracy = correct_total / len(query_dataset)
    print(f'accuracy = {accuracy}')

    if text_total > 0:
        accuracy = (text_total - text_error_total) / text_total
        print(f'text accuracy = {accuracy}')
        
