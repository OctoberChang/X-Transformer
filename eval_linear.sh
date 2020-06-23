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
VERSION=$2

ALGO=5
LABEL_EMB=pifa-neural

PRED_NPZ_PATHS=""
SEED_LIST=( 0 1 2 )
for SEED in "${SEED_LIST[@]}"; do
  # indexer (for reproducibility, use clusters from pretrained_dir)
  OUTPUT_DIR=pretrained_models/${DATASET}/${LABEL_EMB}-a${ALGO}-s${SEED}

  # ranker train and predict
  RANKER_DIR=${OUTPUT_DIR}/ranker_linear-${VERSION}
  PRED_NPZ_PATH=${RANKER_DIR}/tst.pred.npz
  mkdir -p ${RANKER_DIR}

  # x_emb=TF-IDF, model=Parabel
  if [ ${VERSION} == 'v0' ]; then
    python -m xbert.ranker train \
      -x datasets/${DATASET}/X.trn.npz \
      -y datasets/${DATASET}/Y.trn.npz \
      -c ${OUTPUT_DIR}/indexer/code.npz \
      -o ${RANKER_DIR} -t 0.01

    python -m xbert.ranker predict \
      -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
      -x datasets/${DATASET}/X.tst.npz \
      -y datasets/${DATASET}/Y.tst.npz

  # x_emb=xlnet_finetuned+TF-IDF, model=Parabel
  elif [ ${VERSION} == 'v1' ]; then
    python -m xbert.ranker train \
      -x datasets/${DATASET}/X.trn.npz \
      -x2 datasets/${DATASET}/X.trn.xlnet.finetune.npy \
      -y datasets/${DATASET}/Y.trn.npz \
      -c ${OUTPUT_DIR}/indexer/code.npz \
      -o ${RANKER_DIR} -t 0.01 -f 0

    python -m xbert.ranker predict \
      -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
      -x datasets/${DATASET}/X.tst.npz \
      -x2 datasets/${DATASET}/X.tst.xlnet.finetune.npy \
      -y datasets/${DATASET}/Y.tst.npz -f 0
  else
    echo 'unknown linear version'
    exit
  fi

  # append
  PRED_NPZ_PATHS="${PRED_NPZ_PATHS} ${PRED_NPZ_PATH}"
done

# final eval
EVAL_DIR=results_linear
mkdir -p ${EVAL_DIR}
python -u -m xbert.evaluator -y datasets/${DATASET}/Y.tst.npz -e -p ${PRED_NPZ_PATHS} |& tee ${EVAL_DIR}/${DATASET}.${VERSION}.txt

