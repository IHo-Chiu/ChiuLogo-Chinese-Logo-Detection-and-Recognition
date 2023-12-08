import os
import torch
import numpy as np
from PIL import Image
from LogoModel import LogoModel

if __name__ == "__main__":
    model_path = "best.ckpt"
    image_path = "test.jpg"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LogoModel.load_from_checkpoint(model_path).to(device)
    model.eval()
    
    image = Image.open(image_path)
    image = model.preprocess(image)
    
    with torch.no_grad():
        features = model(images)