import os
import random
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from LogoDataset import LogoDataset
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from LogoModel import LogoModel

if __name__ == '__main__':
    model_path = "best.ckpt"
    dataset_dir = 'dataset/LogoDet-3K_preprocessed'
    # dataset_dir = 'dataset/openlogo_preprocessed'
    dataset_split = 'test'
    gallery_dataset_type = 'gallery'
    query_dataset_type = 'query'
    batch_size=128
    num_workers=8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset
    random.seed(5201314)
    torch.manual_seed(5201314)

    # CLIP
    clip_model = LogoModel.load_from_checkpoint(model_path).to(device)
    preprocess = clip_model.preprocess
    clip_model.eval()
    
    gallery_dataset = LogoDataset(dir=dataset_dir, split=dataset_split, type=gallery_dataset_type, transform=preprocess)
    gallery_dataloader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                                  collate_fn=gallery_dataset.collate_fn, pin_memory=True, drop_last=False)

    query_dataset = LogoDataset(dir=dataset_dir, split=dataset_split, type=query_dataset_type, transform=preprocess)
    query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                                    collate_fn=query_dataset.collate_fn, pin_memory=True, drop_last=False)

    gallery_features = []
    gallery_labels = []
    query_features = []
    query_labels = []
    
    with torch.no_grad():
        for data in tqdm(gallery_dataloader):
            labels, images, image_paths = data
            
            labels = labels.to(device)
            images = images.to(device)

            features = clip_model(images)
            
            gallery_features.append(features)
            gallery_labels.append(labels.unsqueeze(1))
    
    gallery_features = torch.vstack(gallery_features).cuda()
    gallery_labels = torch.vstack(gallery_labels).squeeze().float().cuda()
    
    with torch.no_grad():
        for data in tqdm(query_dataloader):
            labels, images, image_paths = data
            
            labels = labels.to(device)
            images = images.to(device)

            features = clip_model(images)

            query_features.append(features)
            query_labels.append(labels.unsqueeze(1))

    query_features = torch.vstack(query_features).cuda()
    query_labels = torch.vstack(query_labels).squeeze().float().cuda()

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    print("\n-------- Evaluating on the test set. --------")

    beta = clip_model.search_beta(gallery_features, gallery_labels, query_features, query_labels)
    predict_labels, confidences = clip_model.predict(query_features, gallery_features, gallery_labels, beta)

    correct = np.sum(predict_labels.cpu().numpy() == query_labels.cpu().numpy())
    acc = 100 * correct / query_labels.shape[0]
    
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))
