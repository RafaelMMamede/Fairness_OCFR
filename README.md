GitHub repository for the Paper: 


Fairness Under Cover: Evaluating the Impact of Occlusions on Demographic Bias in Facial Recognition


Important aspects of the repository:

 - Occlusions.zip contains the occlusions that can be added to the RFW test dataset for RFW1 and RFW4
 - ./models/ contains the used models (B34 - ./models/253682backbone.pth; G34 - ./models/417378backbone.pth; B50 - iresnet50_../models/253682backbone(1).pth; G50 - ./models/417378backbone(1).pth)
 - ./src/ contains the executables used (auxiliary.py - collection of auxiliary functions; fairness_metrics.py - auxiliary file with implementation of fairness metrics; extract_similarities.py - produces a json with the cossine similarity results a .npy file with the true labels in each scenario; extract_metrics.py - extracts the fairness metrics; extract_xSSAB.py was used for extracting the importance maps; extract_overlaps.py calculates the overlap of importance pixels and the occlusion masks)
 - enviornment.yml contains the yml file of the used environment. 


Replication Procedure for RFW1 and RFW4 (WIP):

