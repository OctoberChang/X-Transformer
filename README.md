# Taming Pretrained Transformers for XMC problems

This is a README for the experimental code of the following paper
>[Taming Pretrained Transformers for eXtreme Multi-label Text Classification](https://arxiv.org/abs/1905.02331)

>Wei-Cheng Chang, Hsiang-Fu Yu, Kai Zhong, Yiming Yang, Inderjit Dhillon

>KDD 2020

## Upates (2021-04-27)
Latest implementation (faster training with stronger performance) of X-Transformer is available at [PECOS](https://github.com/amzn/pecos), feel free to try it out!


## Installation
 
### Depedencies via Conda Environment
	
	> conda env create -f environment.yml
	> source activate pt1.2_xmlc_transformer
	> (pt1.2_xmlc_transformer) pip install -e .
	> (pt1.2_xmlc_transformer) python setup.py install --force
	
	
**Notice: the following examples are executed under the ```> (pt1.2_xmlc_transformer)``` conda virtual environment


## Reproduce Evaulation Results in the Paper
We demonstrate how to reproduce the evaluation results in our paper
by downloading the raw dataset and pretrained models.

### Download Dataset (Eurlex-4K, Wiki10-31K, AmazonCat-13K, Wiki-500K)
Change directory into ./datasets folder, download and unzip each dataset

```bash
cd ./datasets
bash download-data.sh Eurlex-4K
bash download-data.sh Wiki10-31K
bash download-data.sh AmazonCat-13K
bash download-data.sh Wiki-500K
cd ../
```

Each dataset contains the following files
- ```label_map.txt```: each line is the raw text of the label
- ```train_raw_text.txt, test_raw_text.txt```: each line is the raw text of the instance
- ```X.trn.npz, X.tst.npz```: instance's embedding matrix (either sparse TF-IDF or fine-tuned dense embedding)  
- ```Y.trn.npz, Y.tst.npz```: instance-to-label assignment matrix
  
### Download Pretrained Models (processed data, Indexing codes, fine-tuned Transformer models)
Change directory into ./pretrained_models folder, download and unzip models for each dataset
	
```bash
cd ./pretrained_models
bash download-models.sh Eurlex-4K
bash download-models.sh Wiki10-31K
bash download-models.sh AmazonCat-13K
bash download-models.sh Wiki-500K
cd ../
```
Each folder has the following strcture
- ```proc_data```: a sub-folder containing: X.{trn|tst}.{model}.128.pkl, C.{label-emb}.npz, L.{label-emb}.npz
- ```pifa-tfidf-s0```: a sub-folder containing indexer and matcher
- ```pifa-neural-s0```: a sub-folder containing indexer and matcher 
- ```text-emb-s0```: a sub-folder containing indexer and matcher


### Evaluate Linear Models
Given the provided indexing codes (label-to-cluster assignments), train/predict linear models, and evaluate with Precision/Recall@k:

```bash
bash eval_linear.sh ${DATASET} ${VERSION}
```

- ```DATASET```: the dataset name such as Eurlex-4K, Wiki10-31K, AmazonCat-13K, or Wiki-500K.
- ```VERSION```: v0=sparse TF-IDF features. v1=sparse TF-IDF features concatenate with dense fine-tuned XLNet embedding.	

The evaluaiton results should located at
``` ./results_linear/${DATASET}.${VERSION}.txt ```


### Evaluate Fine-tuned X-Transformer Models
Given the provided indexing codes (label-to-cluster assignments) and the fine-tuned Transformer models, train/predict ranker of the X-Transformer framework, and evaluate with Precision/Recall@k:

```bash
bash eval_transformer.sh ${DATASET}
```

- ```DATASET```: the dataset name such as Eurlex-4K, Wiki10-31K, AmazonCat-13K, or Wiki-500K.	

The evaluaiton results should located at
``` ./results_transformer/${DATASET}.final.txt ```


## Running X-Transformer on customized datasets
The X-Transformer framework consists of 9 configurations (3 label-embedding times 3 model-type).
For simplicity, we show you 1 out-of 9 here, using ```LABEL_EMB=pifa-tfidf``` and ```MODEL_TYPE=bert```.

We will use Eurlex-4K as an example. In the ./datasets/Eurlex-4K folder, we assume the following files are provided:

- ```X.trn.npz```: the instance TF-IDF feature matrix for the train set. The data type is scipy.sparse.csr_matrix of size (N_trn, D_tfidf), where N_trn is the number of train instances and D_tfidf is the number of features. 
- ```X.tst.npz```: the instance TF-IDF feature matrix for the test set. The data type is scipy.sparse.csr_matrix of size (N_tst, D_tfidf), where N_tst is the number of test instances and D_tfidf is the number of features.
- ```Y.trn.npz```: the instance-to-label matrix for the train set. The data type is scipy.sparse.csr_matrix of size (N_trn, L), where n_trn is the number of train instances and L is the number of labels. 
- ```Y.tst.npz```: the instance-to-label matrix for the test set. The data type is scipy.sparse.csr_matrix of size (N_tst, L), where n_tst is the number of test instances and L is the number of labels.
- ```train_raw_texts.txt```: The raw text of the train set. 
- ```test_raw_texts.txt```: The raw text of the test set.
- ```label_map.txt```: the label's text description. 

Given those input files, the pipeline can be divided into three stages: Indexer, Matcher, and Ranker. 

### Indexer
In stage 1, we will do the following
- (1) construct label embedding
- (2) perform hierarchical 2-means and output the instance-to-cluster assignment matrix
- (3) preprocess the input and output for training Transformer models.

**TLDR**: we combine and summarize (1),(2),(3) into two scripts: ```run_preprocess_label.sh``` and ```run_preprocess_feat.sh```. See more detailed explaination in the following.


(1) To construct label embedding,
```bash
OUTPUT_DIR=save_models/${DATASET}
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
mkdir -p ${PROC_DATA_DIR}
python -m xbert.preprocess \
    --do_label_embedding \
    -i ${DATA_DIR} \
    -o ${PROC_DATA_DIR} \
    -l ${LABEL_EMB} \
    -x ${LABEL_EMB_INST_PATH}
```

- ```DATA_DIR```: ./datasets/Eurlex-4K
- ```PROC_DATA_DIR```: ./save_models/Eurlex-4K/proc_data
- ```LABEL_EMB```: pifa-tfidf (you can also try text-emb or pifa-neural if you have fine-tuned instance embeddings)
- ```LABEL_EMB_INST_PATH```: ./datasets/Eurlex-4K/X.trn.npz

This should yield ```L.${LABEL_EMB}.npz``` in the ```PROC_DATA_DIR```.

(2) To perform hierarchical 2-means,
```bash
SEED_LIST=( 0 1 2 )
for SEED in "${SEED_LIST[@]}"; do
    LABEL_EMB_NAME=${LABEL_EMB}-s${SEED}
    INDEXER_DIR=${OUTPUT_DIR}/${LABEL_EMB_NAME}/indexer
    python -u -m xbert.indexer \
    python -m xbert.preprocess \
        -i ${PROC_DATA_DIR}/L.${LABEL_EMB}.npz \
        -o ${INDEXER_DIR} --seed ${SEED}
```
This should yield ```code.npz``` in the ```INDEXIER_DIR```.

(3) To preprocess input and output for Transformer models,
```bash
SEED=0
LABEL_EMB_NAME=${LABEL_EMB}-s${SEED}
INDEXER_DIR=${OUTPUT_DIR}/${LABEL_EMB_NAME}/indexer
python -u -m xbert.preprocess \
    --do_proc_label \
    -i ${DATA_DIR} \
    -o ${PROC_DATA_DIR} \
    -l ${LABEL_EMB_NAME} \
    -c ${INDEXER_DIR}/code.npz
```
This should yield the instance-to-cluster matrix ```C.trn.npz``` and ```C.tst.npz``` in the ```PROC_DATA_DIR```.

```bash
OUTPUT_DIR=save_models/${DATASET}
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
python -u -m xbert.preprocess \
    --do_proc_feat \
    -i ${DATA_DIR} \
    -o ${PROC_DATA_DIR} \
    -m ${MODEL_TYPE} \
    -n ${MODEL_NAME} \
    --max_xseq_len ${MAX_XSEQ_LEN} \
    |& tee ${PROC_DATA_DIR}/log.${MODEL_TYPE}.${MAX_XSEQ_LEN}.txt
```
- ```MODEL_TYPE```: bert (or roberta, xlnet)
- ```MODEL_NAME```: bert-large-cased-whole-word-masking (or roberta-large, xlnet-large-cased)
- ```MAX_XSEQ_LEN```: maximum number of tokens, we set to 128

This should yield ```X.trn.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pt``` and ```X.tst.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pt``` in the ```PROC_DATA_DIR```.


### Matcher
In stage 2, we will do the following
- (1) train deep Transformer models to map instances to the induced clusters
- (2) output the predicted cluster scores and fine-tune instance embeddings

**TLDR**: ```run_transformer_train.sh```. See more detailed explaination in the following.


(1) Assume we have 8 Nvidia V100 GPUs. To train the models,
```bash
MODEL_DIR=${OUTPUT_DIR}/${INDEXER_NAME}/matcher/${MODEL_NAME}
mkdir -p ${MODEL_DIR}
```
```python
python -m torch.distributed.launch \
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
```
- ```MODEL_TYPE```: bert (or roberta, xlnet)
- ```MODEL_NAME```: bert-large-cased-whole-word-masking (or roberta-large, xlnet-large-cased)
- ```PER_DEVICE_TRN_BSZ```: 16 if using Nvidia V100 (or set to 8 if using Nvidia 2080Ti)
- ```GRAD_ACCU_STEPS```: 2 if using Nvidia V100 (or set to 4 if using Nvidia 2080Ti) 
- ```MAX_STEPS```: set to 1,000 for Eurlex-4K. Depending on your datasets
- ```WARMUP_STEPS```: set to 1,00 for Eurlex-4K. Depending on your datasets
- ```LEARNING_RATE```: set to 5e-5 for Eurlex-4K. Depending on your datasets
- ```LOGGING_STEPS```: set to 100


(2) To generate predictions and instance embedding,
```bash
GPID=0,1,2,3,4,5,6,7
PER_DEVICE_VAL_BSZ=32
```
```python
CUDA_VISIBLE_DEVICES=${GPID} python -u xbert/transformer.py
    -m ${MODEL_TYPE} -n ${MODEL_NAME} \
    --do_eval -o ${MODEL_DIR} \
    -x_trn ${PROC_DATA_DIR}/X.trn.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pkl \
    -c_trn ${PROC_DATA_DIR}/C.trn.${INDEXER_NAME}.npz \
    -x_tst ${PROC_DATA_DIR}/X.tst.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pkl \
    -c_tst ${PROC_DATA_DIR}/C.tst.${INDEXER_NAME}.npz \
    --per_device_eval_batch_size ${PER_DEVICE_VAL_BSZ}
```
This should yield the following output in the ```MODEL_DIR```
- ```C_trn_pred.npz``` and ```C_tst_pred.npz```: model-predicted cluster scores
- ```trn_embeddings.npy``` and ```tst_embeddings.npy```: fine-tuned instance embeddings

### Ranker
In stage 3, we will do the following
- (1) train linear rankers to map instances and predicted cluster scores to label scores
- (2) output top-k predicted labels

**TLDR**: ```run_transformer_predict.sh```. See more detailed explaination in the following.

(1) To train linear rankers,
```bash
LABEL_NAME=pifa-tfidf-s0
MODEL_NAME=bert-large-cased-whole-word-masking
OUTPUT_DIR=save_models/${DATASET}/${LABEL_NAME}
INDEXER_DIR=${OUTPUT_DIR}/indexer
MATCHER_DIR=${OUTPUT_DIR}/matcher/${MODEL_NAME}
RANKER_DIR=${OUTPUT_DIR}/ranker/${MODEL_NAME}
mkdir -p ${RANKER_DIR}
```
```python
python -m xbert.ranker train \
    -x1 ${DATA_DIR}/X.trn.npz \
    -x2 ${MATCHER_DIR}/trn_embeddings.npy \
    -y ${DATA_DIR}/Y.trn.npz \
    -z ${MATCHER_DIR}/C_trn_pred.npz \
    -c ${INDEXER_DIR}/code.npz \
    -o ${RANKER_DIR} -t 0.01 \
    -f 0 --mode ranker
```

(2) To predict the final top-k labels,
```bash
PRED_NPZ_PATH=${RANKER_DIR}/tst.pred.npz
python -m xbert.ranker predict \
    -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
    -x1 ${DATA_DIR}/X.tst.npz \
    -x2 ${MATCHER_DIR}/tst_embeddings.npy \
    -y ${DATA_DIR}/Y.tst.npz \
    -z ${MATCHER_DIR}/C_tst_pred.npz \
    -f 0 -t noop
```

This should yield the predicted top-k labels tst.pred.npz specified in ```PRED_NPZ_PATH```.



## Acknowledge

Some portions of this repo is borrowed from the following repos:
- [transformers(v2.2.0)](https://github.com/huggingface/transformers)
- [liblinear](https://github.com/cjlin1/liblinear)
- [TRMF](https://github.com/rofuyu/exp-trmf-nips16)
