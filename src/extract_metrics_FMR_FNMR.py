import argparse
import pandas as pd
import numpy as np
import sklearn as sk
import fairlearn as fair   # type: ignore
from fairlearn import metrics # type: ignore
from auxiliary import open_results, EER_th, DAccSTD_th
from fairness_metrics import FDR, GARBE, IR
import logging


def map_models(alias):    
    models = ['iresnet34_../models/253682backbone.pth', 'iresnet34_../models/417378backbone.pth', 'iresnet50_../models/253682backbone(1).pth', 'iresnet50_../models/417378backbone(1).pth']
    models_alias = np.array(['Balanced34', 'Global34', 'Balanced50','Global50'],dtype=str)

    index = int(np.squeeze(np.where(models_alias==alias)))

    return models[index]


def mean_absolute_dispersion(X):
    """
    :param X: Sample of values for which to calculate mad
    """
    n = len(X)

    abs_sums = 0

    for i in range(n):
        for j in range(n):

            abs_sums += np.abs(X[i]-X[j])
    
    G = (abs_sums/(n**2))

    return G


if __name__ == '__main__':


    ### user arguments
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("--threshold", type=str, default = 'DAccStd')
    parse.add_argument("--datasets", type=list, default = ['RFW','RFW0-RFW1','RFW0-RFW4'])
    parse.add_argument("--save_path",type=str,default = '../results/FMR_FNMR.csv')
    parse.add_argument("--logging_path", type=str,default ='./extractmetrics_FMR_FNMR.log')
    args = parse.parse_args()

    logging.basicConfig(filename=args.logging_path,level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    #static variables
    ethnicities = ['African', 'Asian','Caucasian','Indian']
    models = ['iresnet34_../models/253682backbone.pth', 'iresnet34_../models/417378backbone.pth', 'iresnet50_../models/253682backbone(1).pth', 'iresnet50_../models/417378backbone(1).pth']
    models_alias = ['Balanced34', 'Global34', 'Balanced50','Global50']
    Datasets = args.datasets


    #final DF
    Results = pd.DataFrame({'Dataset':[],'Model':[],'African_FMR':[],'Asian_FMR':[],'Caucasian_FMR':[],'Indian_FMR':[],'MAD_FMR':[],'Delta_FMR':[],
    'African_FNMR':[],'Asian_FNMR':[],'Caucasian_FNMR':[],'Indian_FNMR':[],'MAD_FNMR':[],'Delta_FNMR':[]})



    #get accuracies 
    for dataset in Datasets:
        similarities, matches = open_results(dataset)

        for i,model in enumerate(models):
            Accs = []

            #calculate threshold
            match_array = np.array([matches[eth] for eth in ethnicities]).flatten()
            similarity_predictions = np.array([similarities[eth][model] for eth in ethnicities]).flatten()

            if args.threshold == 'DAccStd':

                group_array = np.array([len(matches[eth])*[eth] for eth in ethnicities]).flatten()
                th = DAccSTD_th(match_array,similarity_predictions,group_array)
                
            if args.threshold == 'EER':

                th = EER_th(match_array, similarity_predictions)


            FMRs = []
            FNMRs = []

            for eth in ethnicities:

                pred =np.where(similarities[eth][model]>=th,1,0)


                P = np.where(pred ==1)[0]
                N = P = np.where(pred ==0)[0]

                FN = len(np.squeeze(np.where((pred==0)&(matches[eth]==1))))
                FP = len(np.squeeze(np.where((pred==1)&(matches[eth]==0))))

                FNMRs.append(FN/len(N)) 
                FMRs.append(FP/len(P))
                

            delta_FMRs = np.max(FMRs)-np.min(FMRs)
            delta_FNMRs = np.max(FNMRs)-np.min(FNMRs)

            MAD_FMR = mean_absolute_dispersion(FMRs)
            MAD_FNMR = mean_absolute_dispersion(FNMRs)
            
            Results.loc[len(Results)] = [dataset,models_alias[i], FMRs[0],FMRs[1],FMRs[2],FMRs[3],MAD_FMR,delta_FMRs,
                                         FNMRs[0],FNMRs[1],FNMRs[2],FNMRs[3],MAD_FNMR,delta_FNMRs,] 





    Results.to_csv(args.save_path)