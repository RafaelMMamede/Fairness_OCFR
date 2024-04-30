import os
import argparse
import torch 
import numpy as np
import json 
import sys 
sys.path.append('../../ElasticFace') #REPLACE WITH PATH TO ELASTICFACE
from auxiliary import create_model, preprocess_image



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



if __name__ == '__main__':

    ### user arguments
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("--models", type=list, default=['iresnet34','iresnet34','iresnet50','iresnet50']) 
    parse.add_argument("--backbone_paths", type=list, default=['../models/253682backbone.pth','../models/417378backbone.pth','../models/253682backbone(1).pth','../models/417378backbone(1).pth'])   #must match len with models
    parse.add_argument('--data_path',type=str, default="../../Datasets/RFW/test_aligned/data/African/")
    parse.add_argument("--ethnicity",type=str, default="African")
    parse.add_argument("--dataset_name",type=str, default="RFW")
    parse.add_argument("--pairs_path", type=str, default="../../Work/Datasets/RFW/txts/African/African_pairs.txt") 
    args = parse.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cos.to(device)

    resnets = [create_model(args.models[i],args.backbone_paths[i],device) for i in range(len(args.models))]  #list of models extracted
    
    similarity = {}
    for i in range(len(args.models)):
        similarity[args.models[i]+'_'+args.backbone_paths[i]] = []

    

    with open(args.pairs_path) as f:
        lines = f.readlines()
        
        
        matches = []   #0 for not match, 1 for match

        for line in lines:
            line_list = line.split('\t')
            if '\n' in line_list[-1]:
                line_list[-1] = line_list[-1][:-1]

                
            if len(line_list)==3: #matching faces
                ind = line_list[0]  #individual


                im1 = preprocess_image(args.data_path + ind + '/'+ind+'_000'+line_list[1]+'.jpg').to(device)
                im2 = preprocess_image(args.data_path + ind + '/'+ind+'_000'+line_list[2]+'.jpg').to(device)
                for i in range(len(args.models)):
                    similarity[args.models[i]+'_'+args.backbone_paths[i]].append(np.float64(get_cossine_sim(im1,im2, resnets[i]).cpu().numpy()[0]))
                matches.append(1)


            if len(line_list)==4: #non matching faces
                ind1, ind2 = line_list[0], line_list[2]

                im1 = preprocess_image(args.data_path + ind1 + '/'+ind1+'_000'+line_list[1]+'.jpg').to(device)
                im2 = preprocess_image(args.data_path + ind2 + '/'+ind2+'_000'+line_list[3]+'.jpg').to(device)
                for i in range(len(args.models)):
                    similarity[args.models[i]+'_'+args.backbone_paths[i]].append(np.float64(get_cossine_sim(im1,im2, resnets[i]).cpu().numpy()[0]))
                matches.append(0)

            
    writepath = "../results/"+args.dataset_name+ "/" + args.ethnicity +"/cos_sim.json"

    if not os.path.exists("../results/"+args.dataset_name+ "/" + args.ethnicity):
        os.makedirs("../results/"+args.dataset_name+ "/" + args.ethnicity)
    mode = "w+"


    with open(writepath, mode) as outfile: 
        json.dump(similarity, outfile)
    np.save("../results/"+args.dataset_name+ "/" + args.ethnicity +"/matches.npy" , np.array(matches))