import numpy as np
import argparse
import torch 
from auxiliary import create_model, open_results,DAccSTD_th,EER_th
import cv2
import json 
import os
import tqdm
import scipy as sp


def positive_contributions(importance_map,importance_th = .6):

    pos_importance_map = np.where(importance_map>0,importance_map,0)

    normalized_importance = (pos_importance_map)/(np.max(pos_importance_map)) 

    return  np.where(normalized_importance>importance_th,1,0)
  
def negative_contributions(importance_map,importance_th = .4):

    neg_importance_map = np.where(importance_map<0,-1*importance_map,0)

    normalized_importance = (neg_importance_map)/(np.max(neg_importance_map)) 

    return  np.where(normalized_importance > (1-importance_th),1,0)


def important_contributions(importance_map):
    neg = negative_contributions(importance_map)
    pos = positive_contributions(importance_map)

    return  np.logical_or(neg, pos)


contributions_func_dict = {'positive_contributions':positive_contributions,
                           'negative_contributions':negative_contributions,
                           'important_contributions':important_contributions}


def list_of_strings(arg):
    return arg.split(',')


if __name__ == '__main__':

    ### user arguments
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("--models", type=list, default=['iresnet34','iresnet34','iresnet50','iresnet50'])
    parse.add_argument("--backbone_paths", type=list, default=['../models/253682backbone.pth','../models/417378backbone.pth','../models/253682backbone(1).pth','../models/417378backbone(1).pth'])   #must match len with models
    parse.add_argument("--dataset_name",type=str, default="RFW0_RFW1")
    parse.add_argument("--threshold", type=str, default = 'DAccStd')
    parse.add_argument("--importance_mode", type=str, default="important_contributions")

    parse.add_argument("--pairs_paths",type=list, default= ["../../../Datasets/RFW/txts/African/African_pairs.txt",
                                                            "../../../Datasets/RFW/txts/Asian/Asian_pairs.txt",
                                                            "../../../Datasets/RFW/txts/Caucasian/Caucasian_pairs.txt",
                                                            "../../../Datasets/RFW/txts/Indian/Indian_pairs.txt"])


    parse.add_argument('--occlusion_paths_R',type=list_of_strings, default= "../../../Datasets/RFW_occ_protocolo1/African/_mask/,../../../Datasets/RFW_occ_protocolo1/Asian/_mask/,../../../Datasets/RFW_occ_protocolo1/Caucasian/_mask/,../../../Datasets/RFW_occ_protocolo1/Indian/_mask/") #RIGHT IMAGES

    args = parse.parse_args()



    #static variables
    ethnicities = ['African', 'Asian','Caucasian','Indian']
    models_alias = ['Balanced34', 'Global34', 'Balanced50','Global50']
    models = ['iresnet34_../models/253682backbone.pth', 'iresnet34_../models/417378backbone.pth', 'iresnet50_../models/253682backbone(1).pth', 'iresnet50_../models/417378backbone(1).pth']


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    resnets = [create_model(args.models[i],args.backbone_paths[i],device) for i in range(len(args.models))]  #list of models extracted

    for i,model in enumerate(models):

        #Threshold
        similarities, matches = open_results(args.dataset_name)
        match_array = np.array([matches[eth] for eth in ethnicities]).flatten()
        similarity_predictions = np.array([similarities[eth][models[i]] for eth in ethnicities]).flatten() 

        if args.threshold == 'DAccStd':

            group_array = np.array([len(matches[eth])*[eth] for eth in ethnicities]).flatten()
            th = DAccSTD_th(match_array,similarity_predictions,group_array)
            
        if args.threshold == 'EER':

            th = EER_th(match_array, similarity_predictions)
        


        #create overlap dictionary to have following structure: Dic > Ethnicity > TM/FM/TNM/FNM > Overlaps
        overlap_dic ={}


        for e,eth in enumerate(tqdm.tqdm(ethnicities)):


            #TM, TNM, FM, FNM
            TM, TNM, FM, FNM = [], [], [], []
            


            #ir pelos pares
            with open(args.pairs_paths[e]) as f:
                lines = f.readlines()

                for j,line in enumerate(lines):

                    importance_map = np.load("../results/"+args.dataset_name+ "/" + eth +"/Right_xSSAB/" + models_alias[i]+ "/" + str(j)+'.npy')
                    
                    line_list = line.split('\t')
                    if '\n' in line_list[-1]:
                        line_list[-1] = line_list[-1][:-1]

                    
                    if len(line_list)==3: #matching faces
                        ind = line_list[0]  #individual



                        mask_R = cv2.imread((args.occlusion_paths_R[e] + ind + '/'+ind+'_000'+line_list[2]+'.jpg'),cv2.IMREAD_GRAYSCALE)


                        # Apply adaptive thresholding to create a binary image
                        _, binary_image = cv2.threshold(mask_R, 127, 255, cv2.THRESH_BINARY)


                        # Find contours
                        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # Create an empty mask
                        mask = np.zeros_like(binary_image)

                        # Draw contours on the mask and fill the inside
                        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

                        mask_R = np.where(mask_R>=1,1,0) #binarize




                        importance_map_binarized = contributions_func_dict[args.importance_mode](importance_map)

                        overlap = np.sum(np.multiply(mask_R,importance_map_binarized))/np.sum(importance_map_binarized)

                        if similarities[eth][models[i]][j] >= th:  #TM
                            TM.append(overlap)

                        if similarities[eth][models[i]][j] < th:  #FNM
                            FNM.append(overlap)






                    if len(line_list)==4: #non matching faces
                        ind1, ind2 = line_list[0], line_list[2]

                        mask_R = cv2.imread((args.occlusion_paths_R[e] + ind2 + '/'+ind2+'_000'+line_list[3]+'.jpg'),cv2.IMREAD_GRAYSCALE)
                                                # Apply adaptive thresholding to create a binary image
                        
                        _, binary_image = cv2.threshold(mask_R, 127, 255, cv2.THRESH_BINARY)


                        # Find contours
                        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # Create an empty mask
                        mask = np.zeros_like(binary_image)

                        # Draw contours on the mask and fill the inside
                        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

                        mask_R = np.where(mask_R>=1,1,0) #binarize




                        importance_map_binarized = contributions_func_dict[args.importance_mode](importance_map)

                        overlap = np.sum(np.multiply(mask_R,importance_map_binarized))/np.sum(importance_map_binarized)

                        if similarities[eth][models[i]][j] >= th:  #FM
                            FM.append(overlap)

                        if similarities[eth][models[i]][j] < th:  #TNM
                            TNM.append(overlap)


            overlap_dic[eth] = {}
            overlap_dic[eth]['TM'] = TM
            overlap_dic[eth]['FM'] = FM
            overlap_dic[eth]['TNM'] = TNM
            overlap_dic[eth]['FNM'] = FNM

         
        writepath = "../results/"+args.dataset_name+ "/" + 'xSSAB_Overlaps/' + args.importance_mode + '_' + models_alias[i] + '.json'

        if not os.path.exists("../results/"+args.dataset_name+ "/" + 'xSSAB_Overlaps'):
            os.makedirs("../results/"+args.dataset_name+ "/" + 'xSSAB_Overlaps')
        mode = "w+"

        with open(writepath, mode) as outfile: 
            json.dump(overlap_dic, outfile)



