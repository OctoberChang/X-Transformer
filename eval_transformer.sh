#!/bin/bash

DATASET=$1
DATA_DIR=./datasets/${DATASET}

LABEL_NAME_ARR=( pifa-tfidf-s0 pifa-neural-s0 text-emb-s0 )
MODEL_NAME_ARR=( bert-large-cased-whole-word-masking roberta-large xlnet-large-cased )
EXP_NAME=${DATASET}.final

PRED_NPZ_PATHS=""
for LABEL_NAME in "${LABEL_NAME_ARR[@]}"; do
    OUTPUT_DIR=pretrained_models/${DATASET}/${LABEL_NAME}
    INDEXER_DIR=${OUTPUT_DIR}/indexer
    for MODEL_NAME in "${MODEL_NAME_ARR[@]}"; do
        MATCHER_DIR=${OUTPUT_DIR}/matcher/${MODEL_NAME}
        RANKER_DIR=${OUTPUT_DIR}/ranker/${MODEL_NAME}
        mkdir -p ${RANKER_DIR}
        
        # train linear ranker
        python -m xbert.ranker train \
            -x1 ${DATA_DIR}/X.trn.npz \
            -x2 ${MATCHER_DIR}/trn_embeddings.npy \
            -y datasets/${DATASET}/Y.trn.npz \
            -z ${MATCHER_DIR}/C_trn_pred.npz \
            -c ${OUTPUT_DIR}/indexer/code.npz \
            -o ${RANKER_DIR} -t 0.01 \
            -f 0 -ns 0 --mode ranker \

        # predict final label ranking, using transformer's predicted cluster scores
        PRED_NPZ_PATH=${RANKER_DIR}/tst.pred.npz
        python -m xbert.ranker predict \
            -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
            -x1 datasets/${DATASET}/X.tst.npz \
            -x2 ${MATCHER_DIR}/tst_embeddings.npy \
            -y datasets/${DATASET}/Y.tst.npz \
            -z ${MATCHER_DIR}/C_tst_pred.npz \
            -f 0 -t noop

        # append all prediction path
        PRED_NPZ_PATHS="${PRED_NPZ_PATHS} ${PRED_NPZ_PATH}"
    done
done

# final eval
EVAL_DIR=results_transformer
mkdir -p ${EVAL_DIR}
python -u -m xbert.evaluator \
    -y datasets/${DATASET}/Y.tst.npz \
    -e -p ${PRED_NPZ_PATHS} |& tee ${EVAL_DIR}/${EXP_NAME}.txt

