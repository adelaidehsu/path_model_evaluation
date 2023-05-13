eval "$(conda shell.bash hook)"
conda activate turing-academic

export BASE_PATH=~/turing-academic-UCB-UCSF/
export CHECKPOINT_FILE=${BASE_PATH}/src/tnlr/checkpoints/tnlrv3-base.pt
export MODEL_CONFIG_FILE=${BASE_PATH}/src/tnlr/config/tnlr-base-uncased-config.json
export VOCAB_FILE=${BASE_PATH}/src/tnlr/tokenizer/tnlr-uncased-vocab.txt
export PYTHONPATH=${BASE_PATH}/src/

# This is running the code in a single-GPU setup
python extract_features.py \
    --model_name_or_path $CHECKPOINT_FILE \
    --tokenizer_name ${VOCAB_FILE} \
    --config_name ${MODEL_CONFIG_FILE} --do_lower_case \
    --max_seq_length 2500 --task_name prostate
conda deactivate
