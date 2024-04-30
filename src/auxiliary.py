import os
import argparse
import torch 
from torchvision import transforms
from PIL import Image
import numpy as np
import json 
import sys 
sys.path.append('../../ElasticFace') #REPLACE WITH PATH TO ELASTICFACE
import cv2
from backbones.iresnet import iresnet34, iresnet50


def create_model(model,path,device):
    """
    Creates model from:
    :param model: type of model structure in {'iresnet34','iresnet50'}
    :param path: path to model backbone
    :param device: device on which to run the model
    """


    if model == 'iresnet34':
        loaded_model = iresnet34()
    if model == 'iresnet50':
        loaded_model = iresnet50()
    
    state_dict = torch.load(path)
    loaded_model.load_state_dict(state_dict,strict=True)
    loaded_model = loaded_model.to(device)
    return loaded_model


def preprocess_image(image_path):
    """
    Performs pre-processing operations for a single image:
    :param image_path: path to image
    """
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    input_tensor = input_tensor.to(torch.float)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
    return input_batch