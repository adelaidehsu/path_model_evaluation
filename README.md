# Diagnosing Transformers: Illuminating Feature Spaces for Clinical Decision-Making

## Organization
- `prostate`: contains code for fine-tuning and linear probing, and notebooks for corresponding performance evaluation of the two methods.
- `pathology` and `pathology_turing`: contains code for modeling on BERT variants or TNLR.
- `methods`: contains code for producing structured reports from free-text ones.
- `pyfunctions`: contains code for general utilities.
- `interpretations`: contains codes for feature extractions and notebooks for PC evaluations and feature dynamics.

## Set up environment
(1) Install TNLR repo from source

(2) Download TNLR model checkpoint `tnlrv3-base.pt` following the instrutions in their source repo, and put the checkpoint in `turing/src/tnlr/checkpoints/`

(3) Follow the commands to set up a conda environment called "downgrade".
```bash
conda env create -f environment.yml
conda activate downgrade
```

## Finetune
```bash
cd prostate

#For a single fine-tuning job:
python run_ft.py -model_type {bert|tnlr|biobert|clinical_biobert|pubmed_bert} -run {0|1|2} -task {PrimaryGleason|SecondaryGleason|MarginStatusNone|SeminalVesicleNone}

#For running multiple fine-tuning jobs, consider using a script:
bash batch_ft.sh
```

## Linear Probe
You can freeze the first k layers in a model by specifying `-freeze_layer_count k`.

Note: the feature extraction experiment in the paper requires `-freeze_layer_count 12`.

```bash
cd prostate

#For a single linear-probing job:
python run_linear_probe.py -model_type {bert|tnlr|biobert|clinical_biobert|pubmef_bert} -run {0|1|2} -task {PrimaryGleason|SecondaryGleason|MarginStatusNone|SeminalVesicleNone} -freeze_layer_count {1-12}

#For running multiple linear-probing jobs, consider using a script:
bash batch_lp.sh
```
## Citation
If you use any of the code in your work, please cite:
```bash
@inproceedings{
hsu2024diagnosing,
title={Diagnosing Transformers: Illuminating Feature Spaces for Clinical Decision-Making},
author={Aliyah R. Hsu and Yeshwanth Cherapanamjeri and Briton Park and Tristan Naumann and Anobel Odisho and Bin Yu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=k581sTMyPt}
}
```
