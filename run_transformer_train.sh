#!/bin/bash

GPID=$1
DATASET=$2
MODEL_TYPE=$3
INDEXER_NAME=$4 # pifa-tfidf-s0 ||| pifa-neural-s0 ||| text-emb-s0
if [ ${MODEL_TYPE} == "bert" ]; then
    MODEL_NAME=bert-large-cased-whole-word-masking
elif [ ${MODEL_TYPE} == "roberta" ]; then
    MODEL_NAME=roberta-large
elif [ ${MODEL_TYPE} == "xlnet" ]; then
    MODEL_NAME=xlnet-large-cased
else
    echo "unknown MODEL_TYPE! [ bert | robeta | xlnet ]"
    exit
fi
OUTPUT_DIR=save_models/${DATASET}
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
MAX_XSEQ_LEN=128

# Nvidia 2080Ti (11Gb), fp32
PER_DEVICE_TRN_BSZ=8
PER_DEVICE_VAL_BSZ=16
GRAD_ACCU_STEPS=4

# Nvidia V100 (16Gb), fp32
PER_DEVICE_TRN_BSZ=16
PER_DEVICE_VAL_BSZ=32
GRAD_ACCU_STEPS=2

# set hyper-params by dataset
if [ ${DATASET} == "Eurlex-4K" ]; then
    MAX_STEPS=1000
    WARMUP_STEPS=100
    LOGGING_STEPS=50
    LEARNING_RATE=5e-5
elif [ ${DATASET} == "Wiki10-31K" ]; then
    MAX_STEPS=1400
    WARMUP_STEPS=100
    LOGGING_STEPS=50
    LEARNING_RATE=5e-5
elif [ ${DATASET} == "AmazonCat-13K" ]; then
    MAX_STEPS=20000
    WARMUP_STEPS=2000
    LOGGING_STEPS=100
    LEARNING_RATE=8e-5
elif [ ${DATASET} == "Wiki-500K" ]; then
    MAX_STEPS=80000
    WARMUP_STEPS=1000
    LOGGING_STEPS=100
    LEARNING_RATE=6e-5  # users may need to tune this LEARNING_RATE={2e-5,4e-5,6e-5,8e-5} depending on their CUDA/Pytorch environments
else
    echo "dataset not support [ Eurlex-4K | Wiki10-31K | AmazonCat-13K | Wiki-500K ]"
    exit
fi

MODEL_DIR=${OUTPUT_DIR}/${INDEXER_NAME}/matcher/${MODEL_NAME}
mkdir -p ${MODEL_DIR}


# train
CUDA_VISIBLE_DEVICES=${GPID} python -m torch.distributed.launch \
    --nproc_per_node 8 xbert/transformer.py \
    -m ${MODEL_TYPE} -n ${MODEL_NAME} --do_train \
    -x_trn ${PROC_DATA_DIR}/X.trn.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pkl \
    -c_trn ${PROC_DATA_DIR}/C.trn.${INDEXER_NAME}.npz \
    -o ${MODEL_DIR} --overwrite_output_dir \
    --per_device_train_batch_size ${PER_DEVICE_TRN_BSZ} \
    --gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
    --max_steps ${MAX_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --logging_steps ${LOGGING_STEPS} \
    |& tee ${MODEL_DIR}/log.txt


# predict
CUDA_VISIBLE_DEVICES=${GPID} python -u xbert/transformer.py \
    -m ${MODEL_TYPE} -n ${MODEL_NAME} \
    --do_eval -o ${MODEL_DIR} \
    -x_trn ${PROC_DATA_DIR}/X.trn.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pkl \
    -c_trn ${PROC_DATA_DIR}/C.trn.${INDEXER_NAME}.npz \
    -x_tst ${PROC_DATA_DIR}/X.tst.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pkl \
    -c_tst ${PROC_DATA_DIR}/C.tst.${INDEXER_NAME}.npz \
    --per_device_eval_batch_size ${PER_DEVICE_VAL_BSZ}

#### end ####

