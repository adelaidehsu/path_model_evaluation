#! /bin/bash

export CUDA_VISIBLE_DEVICES=1

python run_linear_probe.py -model_type pubmed_bert -run 0 -freeze_layer_count 3 -task SecondaryGleason
python run_linear_probe.py -model_type clinical_biobert -run 0 -freeze_layer_count 3 -task SecondaryGleason
python run_linear_probe.py -model_type biobert -run 0 -freeze_layer_count 3 -task SecondaryGleason
python run_linear_probe.py -model_type tnlr -run 0 -freeze_layer_count 3 -task SecondaryGleason

for d in {4,5,6,7,8,9,10,11}
do
    python run_linear_probe.py -model_type bert -run 0 -freeze_layer_count $d -task SecondaryGleason
    python run_linear_probe.py -model_type pubmed_bert -run 0 -freeze_layer_count $d -task SecondaryGleason
    python run_linear_probe.py -model_type clinical_biobert -run 0 -freeze_layer_count $d -task SecondaryGleason
    python run_linear_probe.py -model_type biobert -run 0 -freeze_layer_count $d -task SecondaryGleason
    python run_linear_probe.py -model_type tnlr -run 0 -freeze_layer_count $d -task SecondaryGleason
done