python extract_overlaps.py
python extract_overlaps.py --importance_mode positive_contributions
python extract_overlaps.py --importance_mode negative_contributions

python extract_overlaps.py --dataset_name RFW0-RFW4 --occlusion_paths_R ../Occlusions/RFW4_occlusions_centered/African/,../Occlusions/RFW4_occlusions_centered/Asian/,../Occlusions/RFW4_occlusions_centered/Caucasian/,../Occlusions/RFW4_occlusions_centered/Indian/
python extract_overlaps.py --dataset_name RFW0-RFW4 --importance_mode positive_contributions --occlusion_paths_R ../Occlusions/RFW4_occlusions_centered/African/,../Occlusions/RFW4_occlusions_centered/Asian/,../Occlusions/RFW4_occlusions_centered/Caucasian/,../Occlusions/RFW4_occlusions_centered/Indian/
python extract_overlaps.py --dataset_name RFW0-RFW4 --importance_mode negative_contributions --occlusion_paths_R ../Occlusions/RFW4_occlusions_centered/African/,../Occlusions/RFW4_occlusions_centered/Asian/,../Occlusions/RFW4_occlusions_centered/Caucasian/,../Occlusions/RFW4_occlusions_centered/Indian/

python extract_overlap_table.py
