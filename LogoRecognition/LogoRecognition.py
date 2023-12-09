import os
import cv2
import torch
import numpy as np
import pickle
from PIL import Image
import ChineseCLIP.cn_clip.clip as clip
from ChineseCLIP.cn_clip.clip import load_from_name, available_models
from ultralytics import YOLO
from LogoModel import LogoModel

class LogoRecognition:
    def __init__(self, pickle_path='latent_space.pkl', device='cuda', detection_model_path='../LogoDetection/yolov8x_logo_detection.pt', recognition_model_path='chiu_logo_recognition.ckpt', conf=0.001, iou=0.1):
        self.pickle_path = pickle_path
        self.latent_space = {'label_names': [], 
                             'labels': torch.FloatTensor([]).to(device), 
                             'embeddings': torch.FloatTensor([]).to(device)}
        self.detection_model = None
        self.recognition_model = None
        self.preprocess = None
        self.device = device
        self.detection_model_path = detection_model_path
        self.recognition_model_path = recognition_model_path
        self.conf = conf
        self.iou = iou

        self.load_pickle()
        self.load_model()

    def load_pickle(self):
        if os.path.exists(self.pickle_path):
            file = open(self.pickle_path, 'rb')
            self.latent_space = pickle.load(file)
            file.close()

    def dump_pickle(self):
        file = open(self.pickle_path, 'wb')
        pickle.dump(self.latent_space, file)
        file.close()

    def load_model(self):
        self.detection_model = YOLO(self.detection_model_path)
        download_root = os.path.dirname(self.recognition_model_path)
        self.recognition_model = LogoModel.load_from_checkpoint(self.recognition_model_path).to(self.device)
        self.recognition_model.eval()
        self.preprocess = self.recognition_model.preprocess

    def run_model(self, img):
        with torch.no_grad():
            result = self.detection_model(img, conf=self.conf, iou=self.iou, device=self.device)[0]
            logo_results = {'bboxes': [], 'detection_confidences': [], 'recognition_results': [], 'recognition_confidences': []}
            bboxes = result.boxes
            w, h, c = img.shape
            crop_imgs = []
            for i in range(len(bboxes)):
                logo = {}
                box = bboxes[i].xyxy[0].detach().cpu().numpy().astype(int)
                if int(box[1]) == int(box[3]) or int(box[0]) == int(box[2]):
                    continue
                crop_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                crop_img = self.preprocess(Image.fromarray(crop_img))
                crop_imgs.append(crop_img)
                logo_results['bboxes'].append(bboxes[i].xyxy[0].detach().cpu().numpy().astype(int).tolist())
                logo_results['detection_confidences'].append(bboxes.conf[i].detach().cpu().item())

            crop_imgs = torch.stack(crop_imgs).to(self.device)
            embeddings = self.recognition_model(crop_imgs)
            labels, confidences = self.recognition_model.predict(embeddings, self.latent_space['embeddings'], self.latent_space['labels'])
            label_names = np.array(self.latent_space['label_names'])[labels.int().cpu()].tolist()
            logo_results['recognition_results'] = label_names
            logo_results['recognition_confidences'] = confidences.detach().cpu().numpy().tolist()
        return logo_results

    def add_logo(self, img, label_name):
        img = self.preprocess(Image.fromarray(img)).unsqueeze(0).to(self.device)
        embedding = self.recognition_model(img)

        self.latent_space['embeddings'] = torch.cat([embedding, self.latent_space['embeddings']], dim=0)

        if label_name not in self.latent_space['label_names']:
            self.latent_space['label_names'].append(label_name)
            
        label = self.latent_space['label_names'].index(label_name)
        label = torch.FloatTensor([label]).to(self.device)
        self.latent_space['labels'] = torch.cat([label, self.latent_space['labels']], dim=0)
        
        assert len(self.latent_space['embeddings']) == len(self.latent_space['labels'])
        self.dump_pickle()

    def draw_on(self, img, results):
        dimg = img.copy()
        for i in range(len(results['bboxes'])):
            box = results['bboxes'][i]
            detection_confidence = results['recognition_confidences'][i]
            recognition_result = results['recognition_results'][i]
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(dimg,'%s,%d%s'%(recognition_result, detection_confidence*100, '%'), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

        return dimg