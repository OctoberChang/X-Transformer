#!/bin/bash

# set DEPTH
if [ $1 == 'Eurlex-4K' ]; then
  DEPTH=6
elif [ $1 == 'Wiki10-31K' ]; then
  DEPTH=9
elif [ $1 == 'AmazonCat-13K' ]; then
  DEPTH=8
elif [ $1 == 'Wiki-500K' ]; then
  DEPTH=13
else
  echo "unknown dataset for the experiment!"
  exit
fi

DATASET=$1

EXP_NAME=feat-joint_neg-yes_noop
PRED_NPZ_PATHS=""
LABEL_NAME_ARR=( pifa-a5-s0 pifa-neural-a5-s0 text-emb-a5-s0 )
MODEL_NAME_ARR=( bert-large-cased-whole-word-masking_seq-128 roberta-large_seq-128 xlnet-large-cased_seq-128 )

for LABEL_NAME in "${LABEL_NAME_ARR[@]}"; do
  OUTPUT_DIR=pretrained_models/${DATASET}/${LABEL_NAME}
  for MODEL_NAME in "${MODEL_NAME_ARR[@]}"; do

    RANKER_DIR=${OUTPUT_DIR}/ranker_${MODEL_NAME}
    mkdir -p ${RANKER_DIR}
    python -m xbert.ranker train \
      -x1 datasets/${DATASET}/X.trn.npz \
      -x2 ${OUTPUT_DIR}/matcher-cased_fp32/${MODEL_NAME}/final_model/trn_embeddings.npy \
      -y datasets/${DATASET}/Y.trn.npz \
      -z ${OUTPUT_DIR}/matcher-cased_fp32/${MODEL_NAME}/final_model/C_trn_pred.npz \
      -c ${OUTPUT_DIR}/indexer/code.npz \
      -o ${RANKER_DIR} -t 0.01 \
      -f 0 -ns 0 --mode ranker \
    
    PRED_NPZ_PATH=${RANKER_DIR}/tst.pred.npz
    python -m xbert.ranker predict \
      -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
      -x1 datasets/${DATASET}/X.tst.npz \
      -x2 ${OUTPUT_DIR}/matcher-cased_fp32/${MODEL_NAME}/final_model/tst_embeddings.npy \
      -y datasets/${DATASET}/Y.tst.npz \
      -z ${OUTPUT_DIR}/matcher-cased_fp32/${MODEL_NAME}/final_model/C_tst_pred.npz -t noop \
      -f 0
    
    PRED_NPZ_PATHS="${PRED_NPZ_PATHS} ${PRED_NPZ_PATH}"
  done
done

# final eval
EVAL_DIR=results_transformer-large/${DATASET}
mkdir -p ${EVAL_DIR}
python -u -m xbert.evaluator -y datasets/${DATASET}/Y.tst.npz -e -p ${PRED_NPZ_PATHS} |& tee ${EVAL_DIR}/${EXP_NAME}.txt
