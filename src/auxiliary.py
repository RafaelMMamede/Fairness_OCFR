import torch 
from torchvision import transforms
from PIL import Image
import numpy as np
import json 
import sys 
from scipy.optimize import brentq, dual_annealing
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import sklearn as sk
import logging


sys.path.append('../../ElasticFace') #REPLACE WITH PATH TO ELASTICFACE
from backbones.iresnet import iresnet34, iresnet50 # type: ignore

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



def open_results(dataset):
    """
    Get dictionary with similarities and array of matches
    :param dataset (str): Dataset from which to get results. From RFW, RFW_unaligned, RFW_protocolo1, RFW_protocolo1_unaligned, RFW_protocolo4, RFW_protocolo4_unaligned
    """
    ethnicities = ['African', 'Asian','Caucasian','Indian']
    similarities = {}
    matches = {}


    for eth in ethnicities:
        with  open('../results/' + dataset + '/' + eth + '/cos_sim.json') as user_file:
            similarities[eth] = json.load(user_file)
        matches[eth] = np.load('../results/' + dataset + '/' + eth + '/matches.npy')

    return similarities, matches



def EER_acc(matches,similarity_prediction):
    """
    Get Accuracy @ Equal Error Rate Threshold 
    :param matches:  true labels
    :param similarity_predictions: cossine similarity between embeddings
    """

    fpr, tpr, thresholds = roc_curve(matches, similarity_prediction, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    pred_label = np.where(similarity_prediction>=thresh,1,0)
    acc = sk.metrics.accuracy_score(matches, pred_label)
    return acc



def EER_th(matches,similarity_prediction):
    """
    Get Equal Error Rate Threshold 
    :param matches:  true labels
    :param similarity_predictions: cossine similarity between embeddings
    """

    fpr, tpr, thresholds = roc_curve(matches, similarity_prediction, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    thresh = interp1d(fpr, thresholds)(eer)
    return thresh


def DAccSTD(th,matches,similarity_prediction,group_list):
    """
    Get accuracy minus std between groups
    :param th: threshold for classification
    :param matches:  true labels
    :param similarity_predictions: cossine similarity between embeddings
    :param group_list: list with information regarding to which group each prediction is part of
    """
    pred_label = np.where(similarity_prediction>=th,1,0)
    global_acc = sk.metrics.accuracy_score(matches, pred_label)
    
    group_list =np.array(group_list)
    unique_labels = np.unique(group_list)
    group_acc = []

    for label in unique_labels:
        indexes  = np.where(group_list == label)[0]

        group_acc.append(sk.metrics.accuracy_score(matches[indexes], pred_label[indexes]))

    return global_acc - np.std(group_acc)




def DAccSTD_th(matches,similarity_prediction,group_list):
    """
    Get Threshold that maximizes accuracy minus std between groups. Requires DAccSTD.
    :param matches:  true labels
    :param similarity_predictions: cossine similarity between embeddings
    :param group_list: list with information regarding to which group each prediction is part of
    """
    result = dual_annealing(lambda th: -1*DAccSTD(th, matches=matches,similarity_prediction=similarity_prediction,group_list=group_list), bounds=list(zip([0.], [1.])))
    logging.info(result)
    return result.x