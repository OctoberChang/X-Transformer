#!/usr/bin/env python
# encoding: utf-8

import argparse
from collections import Counter
import itertools
import json
import os
from os import path
import logging
import numpy as np
import pickle
import time
from tqdm import tqdm
import scipy as sp
import scipy.sparse as smat
from sklearn.preprocessing import normalize
import pandas as pd

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig, DistilBertConfig,)),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer,),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
}


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


def run_label_embedding(args):
    label_map_path = "{}/label_map.txt".format(args.input_data_dir)
    id2label = [line.strip() for line in open(label_map_path, 'r', encoding='ISO-8859-1')]
    n_label = len(id2label)

    if args.label_emb_name.startswith('pifa'):
        if args.label_emb_name.startswith('pifa-tfidf'):
            assert args.inst_embedding.endswith(".npz")
            X = smat.load_npz("{}/X.trn.npz".format(args.input_data_dir))
        elif args.label_emb_name.startswith('pifa-neural'):
            assert args.inst_embedding.endswith(".npy")
            X = np.load(args.inst_embedding)
        else:
            raise ValueError("only support .npz or .npy object!")
        Y = smat.load_npz("{}/Y.trn.npz".format(args.input_data_dir))
        logger.info("X {} {} Y {} {}".format(type(X), X.shape, type(Y), Y.shape))

        # create label embedding
        Y_avg = normalize(Y, axis=1, norm="l2")
        label_embedding = smat.csr_matrix(Y_avg.T.dot(X))
        label_embedding = normalize(label_embedding, axis=1, norm="l2")

    elif args.label_emb_name == "text-emb":
        # xlnet-large-cased tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        model = RobertaModel.from_pretrained("roberta-large")
        model = model.to(device)
        model.eval()

        # get label embedding
        label_embedding = []
        for idx in tqdm(range(n_label)):
            inputs = torch.tensor([tokenizer.encode(id2label[idx])])
            inputs = inputs.to(device)
            with torch.no_grad():
                last_hidden_states = model(inputs)[0]  # [1, seq_len, hidden_dim]
                seq_embedding = last_hidden_states.mean(dim=1)
            label_embedding.append(seq_embedding)
        label_embedding = torch.cat(label_embedding, dim=0)
        label_embedding = label_embedding.cpu().numpy()
        label_embedding = smat.csr_matrix(label_embedding)
        label_embedding = normalize(label_embedding, axis=1, norm="l2")

    else:
        raise NotImplementedError("unknown embed_type {}".format(args.embed_type))

    # save label embedding
    logger.info("label_embedding {} {}".format(type(label_embedding), label_embedding.shape))
    label_embedding_path = "{}/L.{}.npz".format(args.output_data_dir, args.label_emb_name)
    smat.save_npz(label_embedding_path, label_embedding)


def load_feat_data(text_path):
    xseqs = pd.read_csv(text_path, header=None, sep='\t').replace(
        r'\n', ' ', regex=True)[0]  # we replaced any newline characters within each "line" here.
    #Note that this is potentially redundant due to the to_list method.
    xseqs = xseqs.apply(lambda x: x.strip())
    xseq_list = xseqs.to_list()
    logger.info(f'Created X_seq list of size {len(xseq_list)}')
    return xseq_list


def proc_feat(
    args,
    input_text_path,
    tokenizer,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True):

    # load raw text feat data
    xseq_list = load_feat_data(input_text_path)

    # convert raw text into tokens, and convert tokens into tok_ids
    # features: List[Dict(key,val)], where key=['inst_idx', 'input_ids', 'attention_mask', 'token_type_ids']
    features, xseq_lens = [], []
    for (inst_idx, xseq) in enumerate(xseq_list):
        if inst_idx % 1000 == 0:
            logger.info("Writing example %d" % (inst_idx))

        # truncate long text by 4096 chars as they will exceed max_seq_len anyway
        inputs = tokenizer.encode_plus(
            text=xseq[:args.max_trunc_char],
            text_pair=None,
            add_special_tokens=True,
            max_length=args.max_xseq_len,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        xseq_lens.append(len(input_ids))

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_xseq_len - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # sanity check and logging
        assert len(input_ids) == args.max_xseq_len, "Error with input length {} vs {}".format(len(input_ids), args.max_xseq_len)
        assert len(attention_mask) == args.max_xseq_len, "Error with input length {} vs {}".format(len(attention_mask), args.max_xseq_len)
        assert len(token_type_ids) ==args.max_xseq_len, "Error with input length {} vs {}".format(len(token_type_ids), args.max_xseq_len)
        if inst_idx < 5:
            logger.info("*** Example ***")
            logger.info("inst_idx: %s" % (inst_idx))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))

        cur_inst_dict = {
            'inst_idx': inst_idx,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
        features.append(cur_inst_dict)
    # end for loop
    return features, xseq_lens


def main(args):

    if args.do_label_embedding:
        run_label_embedding(args)

    elif args.do_proc_feat:
        # load pretrained model tokenizers
        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        # process train features
        inp_trn_feat_path = os.path.join(args.input_data_dir, 'train_raw_texts.txt')
        logger.info("processing train features {}".format(inp_trn_feat_path))
        trn_features, trn_xseq_lens = proc_feat(
            args, inp_trn_feat_path, tokenizer,
            pad_on_left=bool(args.model_type in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        logger.info(
            "trn_xseq: min={} max={} mean={} median={}".format(
                np.min(trn_xseq_lens), np.max(trn_xseq_lens),
                np.mean(trn_xseq_lens), np.median(trn_xseq_lens),)
        )

        # save trn features
        os.makedirs(args.output_data_dir, exist_ok=True)
        out_trn_feat_path = path.join(args.output_data_dir, "X.trn.{}.{}.pkl".format(args.model_type, args.max_xseq_len))
        with open(out_trn_feat_path, "wb") as fout:
            pickle.dump(trn_features, fout, protocol=pickle.HIGHEST_PROTOCOL)

        # process test features
        inp_tst_feat_path = os.path.join(args.input_data_dir, 'test_raw_texts.txt')
        logger.info("processing test features {}".format(inp_tst_feat_path))
        tst_features, tst_xseq_lens = proc_feat(
            args, inp_tst_feat_path, tokenizer,
            pad_on_left=bool(args.model_type in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        logger.info(
            "tst_xseq: min={} max={} mean={} median={}".format(
                np.min(tst_xseq_lens), np.max(tst_xseq_lens),
                np.mean(tst_xseq_lens), np.median(tst_xseq_lens),)
        )

        # save tst features
        out_tst_feat_path = path.join(args.output_data_dir, "X.tst.{}.{}.pkl".format(args.model_type, args.max_xseq_len))
        with open(out_tst_feat_path, "wb") as fout:
            pickle.dump(tst_features, fout, protocol=pickle.HIGHEST_PROTOCOL)

    elif args.do_proc_label:
        # load existing code
        label2cluster_csr = smat.load_npz(args.input_code_path)
        csr_codes = label2cluster_csr.nonzero()[1]

        # load trn label matrix
        inp_trn_label_path = os.path.join(args.input_data_dir, "Y.trn.npz")
        inp_tst_label_path = os.path.join(args.input_data_dir, "Y.tst.npz")
        Y_trn = smat.load_npz(inp_trn_label_path)
        Y_tst = smat.load_npz(inp_tst_label_path)
        assert Y_trn.shape[1] == label2cluster_csr.shape[0]

        # save C_trn and C_tst
        C_trn = Y_trn.dot(label2cluster_csr)
        C_tst = Y_tst.dot(label2cluster_csr)
        logger.info("NUM_LABELS: {}".format(label2cluster_csr.shape[0]))
        logger.info("NUM_CLUSTERS: {}".format(label2cluster_csr.shape[1]))
        logger.info("C_trn: {}".format(C_trn.shape))
        logger.info("C_tst: {}".format(C_tst.shape))

        out_trn_label_path = os.path.join(args.output_data_dir, "C.trn.{}.npz".format(args.label_emb_name))
        out_tst_label_path = os.path.join(args.output_data_dir, "C.tst.{}.npz".format(args.label_emb_name))
        smat.save_npz(out_trn_label_path, C_trn)
        smat.save_npz(out_tst_label_path, C_tst)

    else:
        raise ValueError("one of --do_label_embedding or --do_proc_feat or --do_proc_label must be set!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument(
        "-i",
        "--input-data-dir",
        type=str,
        required=True,
        metavar="DIR",
        default="./datasets/Eurlex-4K",
        help="path to the dataset directory containing train_texts.txt and test_texts.txt",
    )
    parser.add_argument(
        "-o",
        "--output-data-dir",
        type=str,
        required=True,
        metavar="DIR",
        default="./save_models/Eurlex-4K/proc_data",
        help="directory for storing X.[trn|tst].[model-type].[xseq-len].pkl and C.[trn|tst].npz",
    )
    parser.add_argument(
        "--do_proc_feat", action="store_true", help="Set this flag if you are processing features.",
    )
    parser.add_argument(
        "--do_proc_label", action="store_true", help="Set this flag if you are processing cluster labels.",
    )
    parser.add_argument(
        "--do_label_embedding", action="store_true", help="Set this flag if you are constructing label embeddings.",
    )
    # tokenizers
    parser.add_argument(
        "-m", "--model-type", type=str, default="bert", help="preprocess for model-type [bert | xlnet | xlm | roberta]",
    )
    parser.add_argument(
        "-n",
        "--model_name_or_path",
        type=str,
        default="bert-large-cased-whole-word-masking",
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--max_xseq_len",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--max_trunc_char",
        default=4096,
        type=int,
        help="The maximum total number of character extracted from input raw text for fast processing.\n"
        "Should set it to larger than max_xseq_len*avg_char_per_word."
    )
    # label embedding
    parser.add_argument(
        "-l",
        "--label-emb-name",
        type=str,
        default="pifa-tfidf-a5-s0",
        help="pifa-tfidf-a5-s0 | pifa-neural-a5-s0 | text-emb-a5-s0",
    )
    parser.add_argument(
        "-c",
        "--input-code-path",
        type=str,
        metavar="PATH",
        default="./save_models/Eurlex-4K/pifa-tfidf-a5-s0/indexer/code.npz",
        help="path to the npz file of the indexing codes (CSR, nr_labels * nr_codes)",
    )
    parser.add_argument(
        "-x", "--inst_embedding",
        type=str,
        default=None,
        help="instance embedding for PIFA",
    )
    # parse argument
    args = parser.parse_args()
    print(args)
    main(args)
