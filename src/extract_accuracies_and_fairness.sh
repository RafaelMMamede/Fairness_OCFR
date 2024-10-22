### First save the similarity scores

###### RFW0-RFW0
python extract_similarities.py --data_path_L ../RFW/test_aligned/data/African/  --data_path_R ../RFW/test_aligned/data/African/ --pairs_path ../RFW/test/txts/African/African_pairs.txt --ethnicity African --dataset_name RFW
python extract_similarities.py --data_path_L ../RFW/test_aligned/data/Caucasian/  --data_path_R ../RFW/test_aligned/data/Caucasian/ --pairs_path ../RFW/test/txts/Caucasian/Caucasian_pairs.txt --ethnicity Caucasian --dataset_name RFW
python extract_similarities.py --data_path_L ../RFW/test_aligned/data/Asian/  --data_path_R ../RFW/test_aligned/data/Asian/ --pairs_path ../RFW/test/txts/Asian/Asian_pairs.txt --ethnicity Asian --dataset_name RFW
python extract_similarities.py --data_path_L ../RFW/test_aligned/data/Indian/  --data_path_R ../RFW/test_aligned/data/Indian/ --pairs_path ../RFW/test/txts/Indian/Indian_pairs.txt --ethnicity Indian --dataset_name RFW

###### RFW0-RFW1
python extract_similarities.py --data_path_L ../RFW/test_aligned/data/African/  --data_path_R ../RFW1_v2/African/ --pairs_path ../RFW/test/txts/African/African_pairs.txt --ethnicity African --dataset_name RFW0-RFW1
python extract_similarities.py --data_path_L ../RFW/test_aligned/data/Caucasian/  --data_path_R ../RFW1_v2/Caucasian/ --pairs_path ../RFW/test/txts/Caucasian/Caucasian_pairs.txt --ethnicity Caucasian --dataset_name RFW0-RFW1
python extract_similarities.py --data_path_L ../RFW/test_aligned/data/Asian/  --data_path_R ../RFW1_v2/Asian/ --pairs_path ../RFW/test/txts/Asian/Asian_pairs.txt --ethnicity Asian --dataset_name RFW0-RFW1
python extract_similarities.py --data_path_L ../RFW/test_aligned/data/Indian/  --data_path_R ../RFW1_v2/Indian/ --pairs_path ../RFW/test/txts/Indian/Indian_pairs.txt --ethnicity Indian --dataset_name RFW0-RFW1

###### RFW0-RFW4
python extract_similarities.py --data_path_L ../RFW/test_aligned/data/African/  --data_path_R ../RFW4_v2/African/ --pairs_path ../RFW/test/txts/African/African_pairs.txt --ethnicity African --dataset_name RFW0-RFW4
python extract_similarities.py --data_path_L ../RFW/test_aligned/data/Caucasian/  --data_path_R ../RFW4_v2/Caucasian/ --pairs_path ../RFW/test/txts/Caucasian/Caucasian_pairs.txt --ethnicity Caucasian --dataset_name RFW0-RFW4
python extract_similarities.py --data_path_L ../RFW/test_aligned/data/Asian/  --data_path_R ../RFW4_v2/Asian/ --pairs_path ../RFW/test/txts/Asian/Asian_pairs.txt --ethnicity Asian --dataset_name RFW0-RFW4
python extract_similarities.py --data_path_L ../RFW/test_aligned/data/Indian/  --data_path_R ../RFW4_v2/Indian/ --pairs_path ../RFW/test/txts/Indian/Indian_pairs.txt --ethnicity Indian --dataset_name RFW0-RFW4


### Then obtain the metrics

python extract_metrics.py

python extract_metrics_FMR_FNMR.py