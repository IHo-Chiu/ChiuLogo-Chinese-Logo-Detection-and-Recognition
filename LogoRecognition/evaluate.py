import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from LogoDataset import LogoDataset
from torch.utils.data import DataLoader
import ChineseCLIP.cn_clip.clip as clip
from ChineseCLIP.cn_clip.clip import load_from_name, available_models
from sklearn.neighbors import NearestNeighbors

from STR.strhub.data.module import SceneTextDataModule
from STR.strhub.models.utils import load_from_checkpoint, parse_model_args
from ChineseCLIP.cn_clip.clip.model import convert_models_to_fp32

from LogoModel import LogoModel

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
    
if __name__ == "__main__":
    model_path = "best.ckpt"
    dataset_dir = 'dataset/LogoDet-3K_preprocessed'
    dataset_split = 'test'
    gallery_dataset_type = 'gallery'
    query_dataset_type = 'query'
    batch_size=128
    num_workers=8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LogoModel.load_from_checkpoint(model_path).to(device)
    
    gallery_dataset = LogoDataset(dir=dataset_dir, split=dataset_split, type=gallery_dataset_type, transform=model.preprocess)
    gallery_dataloader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                                  collate_fn=gallery_dataset.collate_fn, pin_memory=True, drop_last=False)

    query_dataset = LogoDataset(dir=dataset_dir, split=dataset_split, type=query_dataset_type, transform=model.preprocess)
    query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                                    collate_fn=query_dataset.collate_fn, pin_memory=True, drop_last=False)

    print(f'{gallery_dataset}: {len(gallery_dataset)}')
    print(f'{query_dataset}: {len(query_dataset)}')

    model.eval()
    gallery_images = []
    gallery_features = []
    gallery_labels = []
    
    with torch.no_grad():
        for data in tqdm(gallery_dataloader):
            labels, images, image_paths = data
            
            labels = labels.to(device)
            images = images.to(device)

            features = model(images)

            gallery_images += image_paths
            gallery_features.append(features)
            gallery_labels.append(labels.unsqueeze(1))

    gallery_images = np.array(gallery_images)
    gallery_features = torch.vstack(gallery_features).detach().cpu().numpy()
    gallery_labels = torch.vstack(gallery_labels).detach().cpu().numpy().squeeze()
    
    knn_model = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine', n_jobs=-1)
    knn_model = knn_model.fit(gallery_features)

    correct_total = 0
    text_total = 0
    text_error_total = 0

    with torch.no_grad():
        for data in tqdm(query_dataloader):
            labels, images, image_paths = data
            images = images.to(device)

            features = model(images)
            
            distances, indices = knn_model.kneighbors(features.cpu())

            if gallery_dataset_type == query_dataset_type:
                indices = np.array(indices)[:, 1]
            else:
                indices = np.array(indices)[:, 0]
                
            predict_labels = gallery_labels[indices]
            predict_images = gallery_images[indices]
            labels = np.array(labels)
            
            correct_map = (labels == predict_labels)
            correct = np.sum(correct_map)
            correct_total += correct

            for idx, correct in enumerate(correct_map):
                query_image = image_paths[idx]
                query_label = os.path.basename(os.path.dirname(query_image))

                if '_text' in query_label:
                    text_total += 1
                
                if correct == False:
                    query_image = image_paths[idx]
                    retrieval_image = predict_images[idx]

                    query_label = os.path.basename(os.path.dirname(query_image))
                    query_label_fix = query_label.replace('_text', '')
                    if query_label_fix[-1] in ['1', '2', '3', '4']:
                        query_label_fix = query_label_fix[:-1]

                    retrieval_label = os.path.basename(os.path.dirname(retrieval_image))
                    retrieval_label_fix = retrieval_label.replace('_text', '')
                    if retrieval_label_fix[-1] in ['1', '2', '3', '4']:
                        retrieval_label_fix = retrieval_label_fix[:-1]
                    
                    if query_label_fix == retrieval_label_fix:
                        correct_total += 1
                        break

                    if '_text' in query_label:
                        text_error_total += 1

                    query_name = f'{query_label}_{os.path.basename(query_image)[:-4]}'
                    retrieval_name = f'{retrieval_label}_{os.path.basename(retrieval_image)[:-4]}'
                    
                    save_name = f'incorrect_images/{query_name}_{retrieval_name}.jpg'
                    a = Image.open(query_image).resize((224, 224))
                    b = Image.open(retrieval_image).resize((224, 224))
                    get_concat_h(a, b).save(save_name)
            
    accuracy = correct_total / len(query_dataset)
    print(f'accuracy = {accuracy}')

    if text_total > 0:
        accuracy = (text_total - text_error_total) / text_total
        print(f'text accuracy = {accuracy}')
        
