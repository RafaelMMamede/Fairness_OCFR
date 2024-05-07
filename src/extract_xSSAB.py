import argparse
import torch 
import numpy as np
import os
import sys 
sys.path.append('../../ElasticFace') #REPLACE WITH PATH TO ELASTICFACE
sys.path.append('../../xSSAB') #REPLACE WITH PATH TO xSSAB
from auxiliary import create_model, open_results,DAccSTD_th,EER_th, create_model_extended, preprocess_image, ExtendediResNet34, ExtendediResNet50
from methodology.gradient_calculator import get_gradient # type: ignore
import tqdm




if __name__ == '__main__':
    
    ### user arguments
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("--models", type=list, default=['iresnet34','iresnet34','iresnet50','iresnet50']) 
    parse.add_argument("--backbone_paths", type=list, default=['../models/253682backbone.pth','../models/417378backbone.pth','../models/253682backbone(1).pth','../models/417378backbone(1).pth'])   #must match len with models
    parse.add_argument('--data_path_L',type=str, default="../../../Datasets/RFW/test_aligned/data/African/") #LEFT IMAGE
    parse.add_argument('--data_path_R',type=str, default="../../../Datasets/RFW/test_aligned/data/African/") #RIGHT IMAGE
    parse.add_argument("--ethnicity",type=str, default="African")
    parse.add_argument("--dataset_name",type=str, default="RFW")
    parse.add_argument("--pairs_path", type=str, default="../../../Datasets/RFW/txts/African/African_pairs.txt") 
    parse.add_argument("--threshold", type=str, default = 'DAccStd')
    args = parse.parse_args()

    #static variables
    ethnicities = ['African', 'Asian','Caucasian','Indian']
    models = ['iresnet34_./Models/253682backbone.pth', 'iresnet34_./Models/417378backbone.pth', 'iresnet50_./Models/253682backbone(1).pth', 'iresnet50_./Models/417378backbone(1).pth']
    models_alias = ['Balanced34', 'Global34', 'Balanced50','Global50']


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnets = [create_model(args.models[i],args.backbone_paths[i],device) for i in range(len(args.models))]  #list of models extracted

    extended_resnets = [create_model_extended(args.models[i],args.backbone_paths[i],device) for i in range(len(args.models))]  #list of models extended extracted


    with open(args.pairs_path) as f:
        lines = f.readlines()
        
        
        matches = []   #0 for not match, 1 for match

        for i,model in enumerate(tqdm.tqdm(resnets)):
            model_cos =extended_resnets[i]
            model.eval()
            model_cos.eval()


            #Threshold
            similarities, matches = open_results(args.dataset_name)
            match_array = np.array([matches[eth] for eth in ethnicities]).flatten()
            similarity_predictions = np.array([similarities[eth][models[i]] for eth in ethnicities]).flatten() 

            if args.threshold == 'DAccStd':

                group_array = np.array([len(matches[eth])*[eth] for eth in ethnicities]).flatten()
                th = DAccSTD_th(match_array,similarity_predictions,group_array)
                
            if args.threshold == 'EER':

                th = EER_th(match_array, similarity_predictions)





            for j,line in enumerate(lines):
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

                im1.requires_grad_(True)
                im2.requires_grad_(True)

                gradient_pos = get_gradient(im1, im2, model_cos, model, 1, 'Balanced34', th)
                gradient_neg = get_gradient(im1, im2, model_cos, model, 2, 'Balanced34', th)

                gradient_1_2_pos = np.mean(gradient_pos,axis=2)
                gradient_1_2_neg = np.mean(gradient_neg,axis=2)

                combi = gradient_1_2_pos - gradient_1_2_neg

                if not os.path.exists("../results/"+args.dataset_name+ "/" + args.ethnicity +"/Right_xSSAB/" + models_alias[i] ):
                    os.makedirs("../results/"+args.dataset_name+ "/" + args.ethnicity +"/Right_xSSAB/" + models_alias[i] )

                #save results
                np.save("../results/"+args.dataset_name+ "/" + args.ethnicity +"/Right_xSSAB/" + models_alias[i] + "/" + str(j)+'.npy' ,combi)



