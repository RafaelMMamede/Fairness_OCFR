import numpy as np
import pandas as pd
import scipy
import json


if __name__ == '__main__':

    FNM_neg_df = pd.DataFrame({'Dataset':[], 'Model':[], 'African': [], 'Asian': [],'Caucasian':[],'Indian':[], 'pval':[] })

    for dataset_name in ['RFW0-RFW1', 'RFW0-RFW4']:
        for model_name in ['Balanced34','Balanced50','Global34','Global50']:
            
            line = []
            line.append(dataset_name)
            line.append(model_name)
            # Opening JSON file
            with open(r'../results/'+ dataset_name+'/xSSAB_Overlaps/negative_contributions_'+model_name +'.json') as json_file:
                data = json.load(json_file)


            ethnicities = ['African', 'Asian','Caucasian','Indian']

            for type_match in ['FNM']:
                for eth in ethnicities:
                    line.append(np.mean(data[eth][type_match]))
            
                line.append(scipy.stats.f_oneway(data['African'][type_match],data['Asian'][type_match],data['Caucasian'][type_match],data['Indian'][type_match], axis=0)[1])

            FNM_neg_df.loc[len(FNM_neg_df)] = line

    FNM_neg_df.to_csv('../results/overlaps_FNM_neg.csv')


    FM_pos_df = pd.DataFrame({'Dataset':[], 'Model':[], 'African': [], 'Asian': [],'Caucasian':[],'Indian':[], 'pval':[] })

    for dataset_name in ['RFW0-RFW1', 'RFW0-RFW4']:
        for model_name in ['Balanced34','Balanced50','Global34','Global50']:
            
            line = []
            line.append(dataset_name)
            line.append(model_name)
            # Opening JSON file
            with open(r'../results/'+ dataset_name+'/xSSAB_Overlaps/positive_contributions_'+model_name +'.json') as json_file:
                data = json.load(json_file)


            ethnicities = ['African', 'Asian','Caucasian','Indian']

            for type_match in ['FM']:
                for eth in ethnicities:
                    line.append(np.mean(data[eth][type_match]))
            
                line.append(scipy.stats.f_oneway(data['African'][type_match],data['Asian'][type_match],data['Caucasian'][type_match],data['Indian'][type_match], axis=0)[1])

            FM_pos_df.loc[len(FM_pos_df)] = line


    FM_pos_df.to_csv('../results/overlaps_FM_pos.csv')