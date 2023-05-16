# path_model_evaluation

## Set up environment
(1) Install TNLR repo from source
```bash
git clone https://github.com/ppotash/turing-academic-UCB-UCSF.git

#rename it as `turing`
mv turing-academic-UCB-UCSF turing

#copy `pathology_turing/` into `turing/`
cp -r pathology_turing turing/.

#rename `turing/pathology_turing/` as  `turing/pathology/`
mv turing/pathology_turing turing/pathology
```
(2) Download TNLR model checkpoint `tnlrv3-base.pt` following the instrutions in their source repo, and put the checkpoint in `turing/checkpoints/`

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
