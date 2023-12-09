import torch
import torch.nn as nn
import numpy as np
import lightning.pytorch as pl

import ChineseCLIP.cn_clip.clip as clip
from ChineseCLIP.cn_clip.clip import load_from_name, available_models
from ChineseCLIP.cn_clip.clip.model import convert_models_to_fp32

from STR.strhub.data.module import SceneTextDataModule
from STR.strhub.models.utils import load_from_checkpoint, parse_model_args

from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity

from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F

class LogoModel(pl.LightningModule):
    def __init__(self, device='cuda', lr=(2e-6, 2e-4)):
        super().__init__()
        
        clip, self.preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
        self.clip = clip.visual
        convert_models_to_fp32(self.clip)
        del clip

        ocr_model_path="parseq_best.ckpt"
        ocr = load_from_checkpoint(ocr_model_path, **parse_model_args([])).to(device)
        self.ocr = ocr.encoder
        del ocr
        
        self.clip_lr, self.ocr_lr = lr
        
        self.validation_step_features = []
        self.validation_step_labels = []
        
        num_classes=1915
        self.clip_loss_func = losses.NormalizedSoftmaxLoss(
            distance=CosineSimilarity(), 
            num_classes=num_classes,
            embedding_size=512)
        self.ocr_loss_func = losses.NormalizedSoftmaxLoss(
            distance=CosineSimilarity(), 
            num_classes=num_classes,
            embedding_size=384)
        
        self.automatic_optimization = False

    def clip_forward(self, images):
        image_features = self.clip(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def ocr_forward(self, images):
        text_features = self.ocr(images)
        text_features = text_features[:, 0, :]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def forward(self, images):
        image_features = self.clip_forward(images)
        text_features = self.ocr_forward(images)
        combine_features = torch.concat([image_features, text_features], dim=1)
        combine_features = combine_features / combine_features.norm(dim=-1, keepdim=True)
        return combine_features
        
    def training_step(self, batch, batch_idx):
        labels, images = batch
        clip_optimizer, clip_loss_optimizer, ocr_optimizer, ocr_loss_optimizer = self.optimizers()
        
        image_features = self.clip_forward(images)
        clip_loss = self.clip_loss_func(image_features, labels)

        clip_optimizer.zero_grad()
        clip_loss_optimizer.zero_grad()
        self.manual_backward(clip_loss)
        clip_optimizer.step()
        clip_loss_optimizer.step()

        text_features = self.ocr_forward(images)
        ocr_loss = self.ocr_loss_func(text_features, labels)
        
        ocr_optimizer.zero_grad()
        ocr_loss_optimizer.zero_grad()
        self.manual_backward(ocr_loss)
        ocr_optimizer.step()
        ocr_loss_optimizer.step()

        self.log("clip_loss", clip_loss, sync_dist=True, prog_bar=True)
        self.log("ocr_loss", ocr_loss, sync_dist=True, prog_bar=True)
 
    def validation_step(self, batch, batch_idx):
        labels, images = batch
        embeddings = self.forward(images)
        self.validation_step_features.append(embeddings)
        self.validation_step_labels.append(labels.unsqueeze(1))
        
    def on_validation_epoch_end(self):
        query_features = torch.vstack(self.validation_step_features).detach().cpu().numpy()
        query_labels = torch.vstack(self.validation_step_labels).detach().cpu().numpy().squeeze()

        knn_model = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine', n_jobs=-1)
        knn_model = knn_model.fit(query_features)

        distances, indices = knn_model.kneighbors(query_features)
        
        indices = np.array(indices)[:, 1]
        predict_labels = query_labels[indices]
        correct_total = np.sum(query_labels == predict_labels)

        accuracy = correct_total / len(predict_labels)
        
        self.log("val_acc", accuracy, sync_dist=True, prog_bar=True)
        
        self.validation_step_features.clear()
        self.validation_step_labels.clear()

    def configure_optimizers(self):
        clip_optimizer = torch.optim.AdamW(self.clip.parameters(), lr=self.clip_lr)
        clip_loss_optimizer = torch.optim.AdamW(self.clip_loss_func.parameters(), lr=self.clip_lr)
        ocr_optimizer = torch.optim.AdamW(self.ocr.parameters(), lr=self.ocr_lr)
        ocr_loss_optimizer = torch.optim.AdamW(self.ocr_loss_func.parameters(), lr=self.ocr_lr)
        return clip_optimizer, clip_loss_optimizer, ocr_optimizer, ocr_loss_optimizer

    def predict(self, query_features, gallery_features, gallery_labels, beta=70):
        affinity = query_features @ gallery_features.permute(1, 0)
        affinity = affinity * (affinity < .99)
        tip_logits = ((-1) * (beta - beta * affinity)).exp() @ F.one_hot(gallery_labels.long()).float()
        predict_labels = tip_logits.topk(1, 1, True, True)[1].t()[0]
        confidences = affinity.max(dim=1)[0]
        return predict_labels, confidences

    def search_beta(self, gallery_features, gallery_labels, query_features, query_labels):
        
        search_scale = [1000, 5]
        search_step = [200, 20]
        
        beta_list = [i * (search_scale[0] - 0.1) / search_step[0] + 0.1 for i in range(search_step[0])]
    
        best_acc = 0
        best_beta = 0
    
        for beta in beta_list:
            predict_labels, confidences = self.predict(query_features, gallery_features, gallery_labels, beta)
            correct = np.sum(predict_labels.cpu().numpy() == query_labels.cpu().numpy())
            acc = 100 * correct / query_labels.shape[0]
        
            if acc > best_acc:
                print("New best setting, beta: {:.2f}; accuracy: {:.2f}".format(beta, acc))
                best_acc = acc
                best_beta = beta
    
        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))
    
        return best_beta