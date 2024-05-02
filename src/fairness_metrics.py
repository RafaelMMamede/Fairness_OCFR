import numpy as np

def FDR(y_true,y_out, sensitive_features, threshold,alpha=0.5):
    
    """
    FDR @ Given threshhold (https://arxiv.org/pdf/2402.01472.pdf)
    :param y_true:  true labels
    :param y_out: cossine similarity between embeddings, prediction before thresholding
    :param sensitive_features: sensitive attributes
    :param threshold: threshold to apply for decision
    """
    
    y_pred = np.where(y_out>=threshold,1,0)

    FMR_dic = {}
    FNMR_dic = {}

    unique_sensitive = np.unique(sensitive_features)

    #formulate FMR and FNMR dic
    for sensitive in unique_sensitive:
        
        indexes = np.squeeze(np.where(sensitive_features==sensitive))

        y_pred_demographic = y_pred[indexes]
        y_true_demographic = y_true[indexes]

        FN = len(np.squeeze(np.where((y_pred_demographic==0)&(y_true_demographic==1))))
        FP = len(np.squeeze(np.where((y_pred_demographic==1)&(y_true_demographic==0))))

        FNMR_dic[sensitive] = FN/len(indexes)
        FMR_dic[sensitive] = FP/len(indexes)
    

    A = 0
    B = 0
    
    #Define A&B
    for i in range(len(unique_sensitive)):
        sen1 = unique_sensitive[i]

        for j in range(i,len(unique_sensitive)):
            sen2 = unique_sensitive[j]

            Cand_A = np.abs(FMR_dic[sen1]-FMR_dic[sen2])
            
            if Cand_A > A:
                A = Cand_A


            Cand_B = np.abs(FNMR_dic[sen1]-FNMR_dic[sen2])

            if Cand_B > B:
                B = Cand_B

    return 1 - (alpha*A + (1-alpha)*B)




def IR(y_true,y_out, sensitive_features, threshold,alpha=0.5):
    
    """
    IR @ Given threshhold (https://arxiv.org/pdf/2402.01472.pdf)
    :param y_true:  true labels
    :param y_out: cossine similarity between embeddings, prediction before thresholding
    :param sensitive_features: sensitive attributes
    :param threshold: threshold to apply for decision
    """
    
    y_pred = np.where(y_out>=threshold,1,0)

    FMR_dic = {}
    FNMR_dic = {}

    unique_sensitive = np.unique(sensitive_features)

    #formulate FMR and FNMR dic
    for sensitive in unique_sensitive:
        
        indexes = np.squeeze(np.where(sensitive_features==sensitive))

        y_pred_demographic = y_pred[indexes]
        y_true_demographic = y_true[indexes]

        FN = len(np.squeeze(np.where((y_pred_demographic==0)&(y_true_demographic==1))))
        FP = len(np.squeeze(np.where((y_pred_demographic==1)&(y_true_demographic==0))))

        FNMR_dic[sensitive] = FN/len(indexes)
        FMR_dic[sensitive] = FP/len(indexes)
    



    #Define max_FMR, min_FMR & max_FNMR, min_FNMR
    max_FMR, min_FMR = 0,np.inf
    max_FNMR, min_FNMR = 0,np.inf

    for sensitive in unique_sensitive:

        FMR = FMR_dic[sensitive]
        FNMR = FNMR_dic[sensitive]

        #update FMR
        if FMR > max_FMR:
            max_FMR = FMR
        if FMR < min_FMR:
            min_FMR =FMR
        

        #update FNMR
        if FNMR > max_FNMR:
            max_FNMR = FNMR
        if FNMR < min_FNMR:
            min_FNMR =FNMR
        



    #Define A&B
    A = max_FMR/min_FMR
    B = max_FNMR/min_FNMR
    return (A**alpha)*(B**(1-alpha)) 



def Garbe_Aux_Gini(X):
    """
    Aux Function 4 GARBE. Calculates generic Gini coefficient using a variant that normalizes the upper bound of the sample
    :param X: Sample of values for which to calculate Gini
    """
    mean_x = np.mean(X)
    n = len(X)

    abs_sums = 0

    for i in range(n):
        for j in range(n):

            abs_sums += np.abs(X[i]-X[j])
    
    G = n/(n-1)*(abs_sums/(2*mean_x*n**2))

    return G





def GARBE(y_true,y_out, sensitive_features, threshold,alpha=0.5):
    
    """
    FDR @ Given threshhold (https://arxiv.org/pdf/2402.01472.pdf)
    :param y_true:  true labels
    :param y_out: cossine similarity between embeddings, prediction before thresholding
    :param sensitive_features: sensitive attributes
    :param threshold: threshold to apply for decision
    """

    y_pred = np.where(y_out>=threshold,1,0)

    FMR_list = []
    FNMR_list = []

    unique_sensitive = np.unique(sensitive_features)

    #formulate FMR and FNMR list
    for sensitive in unique_sensitive:
        
        indexes = np.squeeze(np.where(sensitive_features==sensitive))

        y_pred_demographic = y_pred[indexes]
        y_true_demographic = y_true[indexes]

        FN = len(np.squeeze(np.where((y_pred_demographic==0)&(y_true_demographic==1))))
        FP = len(np.squeeze(np.where((y_pred_demographic==1)&(y_true_demographic==0))))

        FNMR_list.append(FN/len(indexes))
        FMR_list.append(FP/len(indexes))

    A = Garbe_Aux_Gini(np.array(FMR_list))
    B = Garbe_Aux_Gini(np.array(FNMR_list))

    return alpha*A + (1-alpha)*B