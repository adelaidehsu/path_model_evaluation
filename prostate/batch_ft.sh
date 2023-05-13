# !/bin/bash

#export CUDA_VISIBLE_DEVICES=1

for d in {0,1,2}
do
    python run_ft.py -model_type tnlr -run $d -task MarginStatusNone
    python run_ft.py -model_type biobert -run $d -task MarginStatusNone
    python run_ft.py -model_type clinical_biobert -run $d -task MarginStatusNone
    python run_ft.py -model_type bert -run $d -task MarginStatusNone
    python run_ft.py -model_type pubmed_bert -run $d -task MarginStatusNone
done
