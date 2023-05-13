# !/bin/bash

export CUDA_VISIBLE_DEVICES=1

python run_feature_inference.py -model_type bert -task SeminalVesicleNone -layer_num 12
python run_feature_inference.py -model_type tnlr -task SeminalVesicleNone -layer_num 12
python run_feature_inference.py -model_type biobert -task SeminalVesicleNone -layer_num 12
python run_feature_inference.py -model_type clinical_biobert -task SeminalVesicleNone -layer_num 12
python run_feature_inference.py -model_type pubmed_bert -task SeminalVesicleNone -layer_num 12

python run_feature_inference.py -model_type bert -task SecondaryGleason -layer_num 12
python run_feature_inference.py -model_type tnlr -task SecondaryGleason -layer_num 12
python run_feature_inference.py -model_type biobert -task SecondaryGleason -layer_num 12
python run_feature_inference.py -model_type clinical_biobert -task SecondaryGleason -layer_num 12
python run_feature_inference.py -model_type pubmed_bert -task SecondaryGleason -layer_num 12

python run_feature_inference.py -model_type bert -task PrimaryGleason -layer_num 12
python run_feature_inference.py -model_type tnlr -task PrimaryGleason -layer_num 12
python run_feature_inference.py -model_type biobert -task PrimaryGleason -layer_num 12
python run_feature_inference.py -model_type clinical_biobert -task PrimaryGleason -layer_num 12
python run_feature_inference.py -model_type pubmed_bert -task PrimaryGleason -layer_num 12

python run_feature_inference.py -model_type bert -task MarginStatusNone -layer_num 12
python run_feature_inference.py -model_type tnlr -task MarginStatusNone -layer_num 12
python run_feature_inference.py -model_type biobert -task MarginStatusNone -layer_num 12
python run_feature_inference.py -model_type clinical_biobert -task MarginStatusNone -layer_num 12
python run_feature_inference.py -model_type pubmed_bert -task MarginStatusNone -layer_num 12