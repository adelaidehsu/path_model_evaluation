# path_model_evaluation

## Download TNLR model

## Set up environment
`conda env create -f environment.yml`

`conda activate downgrade`

## Finetune
`cd prostate`

For a single fine-tuning job:

`python run_ft.py -model_type {bert|tnlr|biobert|clinical_biobert|pubmed_bert} -run {0|1|2} -task {PrimaryGleason|SecondaryGleason|MarginStatusNone|SeminalVesicleNone}`

For running multiple fine-tuning jobs, consider using a script:

`bash batch_ft.sh`

## Linear Probe
`cd prostate`

For a single linear-probing job:

`python run_linear_probe.py -model_type {bert|tnlr|biobert|clinical_biobert|pubmef_bert} -run {0|1|2} -task {PrimaryGleason|SecondaryGleason|MarginStatusNone|SeminalVesicleNone} -freeze_layer_count {1-12}`

Note: the feature extraction experiment in the paper requires `-freeze_layer_count 12`

For running multiple linear-probing jobs, consider using a script:

`bash batch_lp.sh`
