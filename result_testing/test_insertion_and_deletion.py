import os
import argparse
import torch 
import numpy as np
import json 
import sys 
sys.path.append('../../ElasticFace') #REPLACE WITH PATH TO ELASTICFACE
sys.path.append('../src')
from auxiliary import create_model, preprocess_image
import matplotlib.pyplot as plt

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-10)
def get_cossine_sim(im_tensor1,im_tensor2, resnet,cos=cos):
    """
    Gets similarity between images, used for verification:
    :param im_tensor1: first image tensor
    :param im_tensor2: second image tensor
    :param resnet: model used for embedding extraction
    :param cos: cossine similarity function
    """

    with torch.no_grad():
        resnet.eval()
        emb1 = resnet(im_tensor1)
        emb2 = resnet(im_tensor2)
    
    return cos(emb1,emb2)


def insertion(image_tensor,insertion_proportion,device,saliency_map=None):

    w,h = image_tensor.shape[-2],image_tensor.shape[-1]


    if saliency_map is None:
        flattened_insertion = np.concatenate((np.ones(int(w*h*insertion_proportion)),np.zeros(int(w*h*(1-insertion_proportion)))))
        np.random.shuffle(flattened_insertion)

        insertion_mask = np.reshape(flattened_insertion, (w,h))

    else:
        cap = np.sort(saliency_map.flatten)[int(w*h*insertion_proportion)]

        insertion_mask = np.where(saliency_map>=cap,1,0)

    insertion_mask = np.array([insertion_mask]*3) #insertion mask as (3,w,h)
    insertion_mask = torch.from_numpy(insertion_mask).type(torch.FloatTensor).to(device) #to torch & to device
    insertion_mask = insertion_mask.unsqueeze(0)  # Add a batch dimension

    return torch.mul(image_tensor,insertion_mask)





if __name__ == '__main__':

     ### user arguments
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("--mode", type=str, default='insertion') 
    parse.add_argument("--element_index", type=int, default=0)
    parse.add_argument("--proportion_list", type=list, default=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    parse.add_argument("--models", type=list, default=['iresnet34','iresnet34','iresnet50','iresnet50']) 
    parse.add_argument("--backbone_paths", type=list, default=['../models/253682backbone.pth','../models/417378backbone.pth','../models/253682backbone(1).pth','../models/417378backbone(1).pth'])   #must match len with models
    parse.add_argument('--data_path_L',type=str, default="../../../Datasets/RFW/test_aligned/data/African/") #LEFT IMAGE
    parse.add_argument('--data_path_R',type=str, default="../../../Datasets/RFW/test_aligned/data/African/") #RIGHT IMAGE
    parse.add_argument("--ethnicity",type=str, default="African")
    parse.add_argument("--dataset_name",type=str, default="RFW")
    parse.add_argument("--pairs_path", type=str, default="../../../Datasets/RFW/txts/African/African_pairs.txt") 
    args = parse.parse_args()

    #static variables
    models_alias = ['Balanced34', 'Global34', 'Balanced50','Global50']


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cos.to(device)

    resnets = [create_model(args.models[i],args.backbone_paths[i],device) for i in range(len(args.models))]  #list of models extracted
    
    similarity = {}
    for i in range(len(args.models)):
        similarity[args.models[i]+'_'+args.backbone_paths[i]] = []

    

    with open(args.pairs_path) as f:
        lines = f.readlines()
        
        


        line = lines[args.element_index]


        line_list = line.split('\t')
        if '\n' in line_list[-1]:
            line_list[-1] = line_list[-1][:-1]

            
        if len(line_list)==3: #matching faces
            ind = line_list[0]  #individual


            im1 = preprocess_image(args.data_path_L + ind + '/'+ind+'_000'+line_list[1]+'.jpg').to(device)
            im2 = preprocess_image(args.data_path_R + ind + '/'+ind+'_000'+line_list[2]+'.jpg').to(device)


        if len(line_list)==4: #non matching faces
            ind1, ind2 = line_list[0], line_list[2]

            im1 = preprocess_image(args.data_path_L + ind1 + '/'+ind1+'_000'+line_list[1]+'.jpg').to(device)
            im2 = preprocess_image(args.data_path_R + ind2 + '/'+ind2+'_000'+line_list[3]+'.jpg').to(device)


        for i in range(len(args.models)):

                importance_map = np.load("../results/"+args.dataset_name+ "/" + args.ethnicity +"/Right_xSSAB/" + models_alias[i]+ "/" + str(args.element_index)+'.npy')

                if args.mode == 'insertion':

                    score_random = []
                    score_importance = []

                    for insertion_proportion in args.proportion_list:

                        #random insertion
                        random_insertion = insertion(im2,insertion_proportion,device,saliency_map=None)
                        score_random.append(np.float64(get_cossine_sim(im1,random_insertion, resnets[i]).cpu().numpy()[0]))

                        #importance map insertion
                        importance_insertion = insertion(im2,insertion_proportion,device,saliency_map=importance_map)
                        score_importance.append(np.float64(get_cossine_sim(im1,importance_insertion, resnets[i]).cpu().numpy()[0]))

                    plt.plot(args.proportion_list,score_random,label = 'random')
                    plt.plot(args.proportion_list,score_importance,label = 'salency_map')
                    plt.legend()
                    plt.title('Insertion Curves | Dataset ' + args.dataset_name + '| Pair '+ str(args.element_index) + '| Model '+ models_alias[i]  )
                    plt.savefig('./insertion_deletion_plots/insertion'+  args.dataset_name +'_'+ str(args.element_index) +'_'+ models_alias[i] + '.png')



            

