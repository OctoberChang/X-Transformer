#!/bin/bash

DATASET=$1              # Eurlex-4K | Wiki10-31K | AmazonCat-13K | Wiki-500K
LABEL_EMB=$2            # pifa-tfidf | pifa-neural | text-emb

MODEL_NAME=xlnet-large-cased
DATA_DIR=./datasets/${DATASET}
VERSION=${LABEL_EMB}.v1


PRED_NPZ_PATHS=""
SEED_LIST=( 0 1 2 )
for SEED in "${SEED_LIST[@]}"; do
	# ranker train and predict
    OUTPUT_DIR=save_models/${DATASET}/${LABEL_EMB}-s${SEED}
	MATCHER_DIR=save_models/${DATASET}/${LABEL_EMB}-s0/matcher/${MODEL_NAME}
    RANKER_DIR=${OUTPUT_DIR}/ranker/${VERSION}
	mkdir -p ${RANKER_DIR}
	
    python -m xbert.ranker train \
		-x1 ${DATA_DIR}/X.trn.npz \
		-x2 ${MATCHER_DIR}/trn_embeddings.npy \
		-y ${DATA_DIR}/Y.trn.npz \
		-c ${OUTPUT_DIR}/indexer/code.npz \
		-o ${RANKER_DIR} -t 0.01 -f 0

	PRED_NPZ_PATH=${RANKER_DIR}/tst.pred.npz
	python -m xbert.ranker predict \
		-m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
		-x1 ${DATA_DIR}/X.tst.npz \
		-x2 ${MATCHER_DIR}/tst_embeddings.npy \
		-y ${DATA_DIR}/Y.tst.npz -f 0

	# append
	PRED_NPZ_PATHS="${PRED_NPZ_PATHS} ${PRED_NPZ_PATH}"
done

# final eval
EVAL_DIR=results_linear
mkdir -p ${EVAL_DIR}
python -u -m xbert.evaluator \
    -y ${DATA_DIR}/Y.tst.npz \
    -e -p ${PRED_NPZ_PATHS} \
    |& tee ${EVAL_DIR}/${DATASET}.${VERSION}.txt

