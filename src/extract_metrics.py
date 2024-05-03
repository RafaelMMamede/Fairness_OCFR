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
    models = ['iresnet34_./Models/253682backbone.pth', 'iresnet34_./Models/417378backbone.pth', 'iresnet50_./Models/253682backbone(1).pth', 'iresnet50_./Models/417378backbone(1).pth']
    models_alias = np.array(['Balanced34', 'Global34', 'Balanced50','Global50'],dtype=str)

    index = int(np.squeeze(np.where(models_alias==alias)))

    return models[index]



if __name__ == '__main__':


    ### user arguments
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("--threshold", type=str, default = 'DAccStd')
    parse.add_argument("--datasets", type=list, default = ['RFW', 'RFW_protocolo1',  'RFW_protocolo4'])
    parse.add_argument("--save_path",type=str,default = '../results/accuracies_and_fairness.csv')
    parse.add_argument("--loging_path", type=str,default ='./extractmetrics.log')
    args = parse.parse_args()

    logging.basicConfig(filename=args.loging_path,level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    #static variables
    ethnicities = ['African', 'Asian','Caucasian','Indian']
    models = ['iresnet34_./Models/253682backbone.pth', 'iresnet34_./Models/417378backbone.pth', 'iresnet50_./Models/253682backbone(1).pth', 'iresnet50_./Models/417378backbone(1).pth']
    models_alias = ['Balanced34', 'Global34', 'Balanced50','Global50']
    Datasets = args.datasets


    #final DF
    Results = pd.DataFrame({'Dataset':[],'Model':[],'African':[],'Asian':[],'Caucasian':[],'Indian':[]})



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



            for eth in ethnicities:

                pred =np.where(similarities[eth][model]>=th,1,0)
                Accs.append(sk.metrics.accuracy_score(matches[eth],pred)*100)
            
            Results.loc[len(Results)] = [dataset,models_alias[i],Accs[0],Accs[1],Accs[2],Accs[3]] 


    #---------------fairness metrics--------------------

    #STD&&SER

    Results['STD'] = np.std(np.array([Results['African'], Results['Asian'], Results['Caucasian'], Results['Indian']]),axis=0)

    Results['SER'] = np.divide(100 - np.min(np.array([Results['African'], Results['Asian'], Results['Caucasian'], Results['Indian']]),axis=0), 
          100 - np.max(np.array([Results['African'], Results['Asian'], Results['Caucasian'], Results['Indian']]),axis=0))
    



    #Demographic Parity && Equilized Odds Fairness Discrepancy Rate (FDR) && Gini Aggregation Rate for Biometric Equitability(GARBE) && Inequity Rate (IR)


    Eqq_odds_diff = []
    Dem_par_diff = []

    FDRs = []
    IRs = []
    GARBEs = []


    for ind in Results.index:
    
        dataset = Results['Dataset'][ind]

        model = map_models(Results['Model'][ind])


        y_true = []
        y_pred = []
        y_out = []
        sensitive_features = []

        similarities, matches = open_results(dataset)


        #calculate threshold
        match_array = np.array([matches[eth] for eth in ethnicities]).flatten()
        similarity_predictions = np.array([similarities[eth][model] for eth in ethnicities]).flatten()

        if args.threshold == 'DAccStd':

            group_array = np.array([len(matches[eth])*[eth] for eth in ethnicities]).flatten()
            th = DAccSTD_th(match_array,similarity_predictions,group_array)
            
        if args.threshold == 'EER':

            th = EER_th(match_array, similarity_predictions)


        for eth in ethnicities:

            previsoes = np.where(similarities[eth][model]>=th,1,0)

            y_pred.append(previsoes)
            y_out.append(similarities[eth][model])
            sensitive_features += len(previsoes)*[eth]

            y_true.append(matches[eth])

        Dem_par_diff.append(fair.metrics.demographic_parity_difference(y_true=np.array(y_true).flatten(),
                                            y_pred=np.array(y_pred).flatten(),
                                            sensitive_features=np.array(sensitive_features).flatten()))
        Eqq_odds_diff.append(fair.metrics.equalized_odds_difference(y_true=np.array(y_true).flatten(),
                                            y_pred=np.array(y_pred).flatten(),
                                            sensitive_features=np.array(sensitive_features).flatten()))
        
        FDRs.append(FDR(y_true=np.array(y_true).flatten(),
                                        y_out=np.array(y_out).flatten(),
                                        sensitive_features=np.array(sensitive_features).flatten(),threshold=th))

        IRs.append(IR(y_true=np.array(y_true).flatten(),
                                                y_out=np.array(y_out).flatten(),
                                                sensitive_features=np.array(sensitive_features).flatten(),threshold=th))
        
        GARBEs.append(GARBE(y_true=np.array(y_true).flatten(),
                                                y_out=np.array(y_out).flatten(),
                                                sensitive_features=np.array(sensitive_features).flatten(),threshold=th))
    Results['EqOdds(Diff)'] = Eqq_odds_diff
    Results['DemPar(Diff)'] = Dem_par_diff

    
    Results['FDR'] = FDRs
    Results['IR'] = IRs
    Results['GARBE'] = GARBEs


    Results.to_csv(args.save_path)