#!/bin/bash

DATASET=$1
VERSION=$2
LABEL_EMB=pifa-tfidf
DATA_DIR=./datasets/${DATASET}

PRED_NPZ_PATHS=""
SEED_LIST=( 0 1 2 )
for SEED in "${SEED_LIST[@]}"; do
    # indexer (for reproducibility, use clusters from pretrained_dir)
    OUTPUT_DIR=pretrained_models/${DATASET}/${LABEL_EMB}-s${SEED}
    INDEXER_DIR=${OUTPUT_DIR}/indexer
    RANKER_DIR=${OUTPUT_DIR}/ranker/linear-${VERSION}
    mkdir -p ${RANKER_DIR}

    # ranker train and predict
    PRED_NPZ_PATH=${RANKER_DIR}/tst.pred.npz

    # x_emb=TF-IDF, model=Parabel
    if [ ${VERSION} == 'v0' ]; then
        python -m xbert.ranker train \
            -x ${DATA_DIR}/X.trn.npz \
            -y ${DATA_DIR}/Y.trn.npz \
            -c ${INDEXER_DIR}/code.npz \
            -o ${RANKER_DIR} -t 0.01

        python -m xbert.ranker predict \
            -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
            -x ${DATA_DIR}/X.tst.npz \
            -y ${DATA_DIR}/Y.tst.npz

    # x_emb=xlnet_finetuned+TF-IDF, model=Parabel
    elif [ ${VERSION} == 'v1' ]; then
        python -m xbert.ranker train \
            -x ${DATA_DIR}/X.trn.npz \
            -x2 ${DATA_DIR}/X.trn.finetune.xlnet.npy \
            -y ${DATA_DIR}/Y.trn.npz \
            -c ${INDEXER_DIR}/code.npz \
            -o ${RANKER_DIR} -t 0.01 -f 0

        python -m xbert.ranker predict \
            -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
            -x ${DATA_DIR}/X.tst.npz \
            -x2 ${DATA_DIR}/X.tst.finetune.xlnet.npy \
            -y ${DATA_DIR}/Y.tst.npz -f 0

    else
        echo 'unknown linear version'
        exit
    fi

    # append all prediction path
    PRED_NPZ_PATHS="${PRED_NPZ_PATHS} ${PRED_NPZ_PATH}"
done

# final eval
EVAL_DIR=results_linear
mkdir -p ${EVAL_DIR}
python -u -m xbert.evaluator \
    -y datasets/${DATASET}/Y.tst.npz \
    -e -p ${PRED_NPZ_PATHS} \
    |& tee ${EVAL_DIR}/${DATASET}.${VERSION}.txt

