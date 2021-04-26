#!/bin/bash

DATASET=$1
MODEL_TYPE=$2
MAX_XSEQ_LEN=$3

# HuggingFace pretrained model preprocess
if [ $MODEL_TYPE == "bert" ]; then
    MODEL_NAME="bert-large-cased-whole-word-masking"
elif [ $MODEL_TYPE == "roberta" ]; then
    MODEL_NAME="roberta-large"
elif [ $MODEL_TYPE == 'xlnet' ]; then
    MODEL_NAME="xlnet-large-cased"
else
    echo "Unknown MODEL_NAME!"
    exit
fi


OUTPUT_DIR=save_models/${DATASET}
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
mkdir -p ${PROC_DATA_DIR}
python -u -m xbert.preprocess \
    --do_proc_feat \
    -i ./datasets/${DATASET} \
    -o ${PROC_DATA_DIR} \
    -m ${MODEL_TYPE} \
    -n ${MODEL_NAME} \
    --max_xseq_len ${MAX_XSEQ_LEN} \
    |& tee ${PROC_DATA_DIR}/log.${MODEL_TYPE}.${MAX_XSEQ_LEN}.txt
