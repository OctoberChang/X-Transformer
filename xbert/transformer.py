# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import copy
import csv
import glob
import json
import logging
import math
import numpy as np
import os
import random
import re
import pickle
import shutil
import tarfile
import tempfile
import scipy as sp
import scipy.sparse as smat
import sys

import time
from os import path
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange

import xbert.rf_linear as rf_linear
import xbert.rf_util as rf_util
from xbert.modeling import BertForXMLC, RobertaForXMLC, XLNetForXMLC

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLNetConfig,
    XLNetTokenizer,
)

from transformers import AdamW, get_linear_schedule_with_warmup


# global variable within the module

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, XLNetConfig)), (),)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForXMLC, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForXMLC, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForXMLC, XLNetTokenizer),
}

logger = None


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# transform model prediction optimized under margin-loss
# into smoother curve for ranker
def transform_prediction(csr_codes, transform="lpsvm-l2"):
    if transform == "sigmoid":
        csr_codes.data[:] = rf_linear.Transform.sigmoid(csr_codes.data[:])
    elif transform == "lpsvm-l2":
        csr_codes.data[:] = rf_linear.Transform.lpsvm(2, csr_codes.data[:])
    elif transform == "lpsvm-l3":
        csr_codes.data[:] = rf_linear.Transform.lpsvm(3, csr_codes.data[:])
    else:
        raise NotImplementedError("unknown transform {}".format(transform))
    return csr_codes

class HingeLoss(nn.Module):
	"""criterion for loss function
	y: 0/1 ground truth matrix of size: batch_size x output_size
	f: real number pred matrix of size: batch_size x output_size
	"""

	def __init__(self, margin=1.0, squared=True):
		super(HingeLoss, self).__init__()
		self.margin = margin
		self.squared = squared

	def forward(self, f, y, C_pos=1.0, C_neg=1.0):
		# convert y into {-1,1}
		y_new = 2.0 * y - 1.0
		tmp = y_new * f

		# Hinge loss
		loss = F.relu(self.margin - tmp)
		if self.squared:
			loss = loss ** 2
		loss = loss * (C_pos * y + C_neg * (1.0 - y))
		return loss.mean()


class TransformerMatcher(object):
    """ TODO Doc"""

    def __init__(self, model=None, num_clusters=None):
        self.model = model
        self.num_clusters = num_clusters
        self.loss_fn = HingeLoss(margin=1.0, squared=True)

    @staticmethod
    def get_args_and_set_logger():
        global logger
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
        )
        logger = logging.getLogger(__name__)
        parser = argparse.ArgumentParser(description="")

        ## Required parameters
        parser.add_argument(
            "-m", "--model-type", type=str, required=True, default="bert", help="preprocess for model-type [bert | xlnet | xlm | roberta]",
        )
        parser.add_argument(
            "-n",
            "--model_name_or_path",
            type=str,
            required=True,
            default="bert-base-uncased",
            help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
        )
        parser.add_argument(
            "-x_trn",
			"--trn_feat_path",
			default="./save_models/Eurlex-4K/proc_data/X.trn.bert.128.pkl",
            type=str,
        )
        parser.add_argument(
            "-x_tst",
			"--tst_feat_path",
			default="./save_models/Eurlex-4K/proc_data/X.tst.bert.128.pkl",
            type=str,
        )
        parser.add_argument(
            "-c_trn",
			"--trn_label_path",
			default="./save_models/Eurlex-4K/proc_data/C.trn.pifa-tfidf-s0.npz",
            type=str,
        )
        parser.add_argument(
            "-c_tst",
			"--tst_label_path",
			default="./save_models/Eurlex-4K/proc_data/C.tst.pifa-tfidf-s0.npz",
            type=str,
        )
        parser.add_argument(
            "-o",
            "--output_dir",
            default="./tmp",
            type=str,
            help="The output directory where the model predictions and checkpoints will be written.",
        )
        ## Other parameters
        parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
        parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
        parser.add_argument(
            "--hidden_dropout_prob", default=0.1, type=float, help="hidden dropout prob in deep transformer models.",
        )
        parser.add_argument(
            "--per_device_train_batch_size", default=8, type=int, help="Batch size per GPU for training.",
        )
        parser.add_argument(
            "--per_device_eval_batch_size", default=8, type=int, help="Batch size per GPU for evaluation.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument(
            "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.",
        )
        parser.add_argument(
            "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.",
        )
        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        parser.add_argument(
            "--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.",
        )
        parser.add_argument(
            "--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
        )
        parser.add_argument(
            "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.",
        )
        parser.add_argument(
            "--logging_steps", default=100, type=int, help="Log every X updates steps.",
        )
        parser.add_argument(
            "--loss_func", default="l2-hinge", type=str, help="loss function: bce | l1-hinge | l2-hinge",
        )
        parser.add_argument("--margin", default=1.0, type=float, help="margin in hinge loss")
        parser.add_argument(
            "--only_topk", default=10, type=int, help="store topk prediction for matching stage",
        )

        parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
        parser.add_argument(
            "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
        )
        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
        )
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

        parser.add_argument(
            "--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        )
        parser.add_argument(
            "--fp16_opt_level",
            type=str,
            default="O1",
            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html",
        )
        parser.add_argument(
            "--local_rank", type=int, default=-1, help="For distributed training: local_rank",
        )

        args = parser.parse_args()
        return {"parser": parser, "logger": logger, "args": args}


    @staticmethod
    def set_device(args):
        """ set device for multi-gpu training, and fix random seed, and exp logging. """

         # Setup CUDA, GPU & distributed training
        if args.no_cuda:
            device = torch.device("cpu")
            args.n_gpu = 0
        elif args.local_rank == -1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            args.n_gpu = torch.cuda.device_count()
        else:
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", args.local_rank)
            args.n_gpu = 1
        if device.type == "cuda":
            torch.cuda.set_device(device)
        args.device = device

        logger.info(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            args.local_rank,
            args.device,
            args.n_gpu,
            bool(args.local_rank != -1),
            args.fp16,
        )
        # Set seed
        set_seed(args)

    def prepare_model(self, args):
        """ Load a pretrained model for sequence classification. """
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            hidden_dropout_prob=args.hidden_dropout_prob,
            num_labels=self.num_clusters,
            finetuning_task=None,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        model.to(args.device)

        # overwrite
        self.config = config
        self.model = model

    def save_model(self, args):
        # Save model checkpoint
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    def predict(self, args, X_eval, C_eval_true, topk=10, get_hidden=False):
        """Prediction interface"""
        args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
        all_inst_idx = torch.tensor([f["inst_idx"] for f in X_eval], dtype=torch.long)
        all_input_ids = torch.tensor([f["input_ids"] for f in X_eval], dtype=torch.long)
        all_attention_mask = torch.tensor([f["attention_mask"] for f in X_eval], dtype=torch.long)
        all_token_type_ids = torch.tensor([f["token_type_ids"] for f in X_eval], dtype=torch.long)
        eval_data = TensorDataset(all_inst_idx, all_input_ids, all_attention_mask, all_token_type_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        if args.local_rank in [-1, 0]:
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(X_eval))
            logger.info("  Batch size = %d", args.eval_batch_size)

        total_loss = 0.0
        total_example = 0.0
        rows, cols, vals = [], [], []
        all_pooled_output = []
        self.model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                inst_idx = batch[0]
                inputs = {
                    "input_ids": batch[1].to(args.device),
                    "attention_mask": batch[2].to(args.device),
                }
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[3].to(args.device) if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM, DistilBERT and RoBERTa don't use segment_ids
                cur_batch_size = inputs["input_ids"].shape[0]

                # forward
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                )
                if get_hidden and self.config.output_hidden_states:
                    c_pred, hidden_states = outputs[0], outputs[1]
                else:
                    c_pred = outputs[0]

                # compute loss
                c_eval = np.array(C_eval_true[inst_idx].toarray())
                c_eval = torch.tensor(c_eval, dtype=torch.float).to(args.device)
                loss = self.loss_fn(c_pred, c_eval)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                total_loss += cur_batch_size * loss

                # get pooled_output, which is the [CLS] embedding for the document
                if get_hidden:
                    if args.model_type == "bert":
                        if args.n_gpu > 1:
                         # assume self.model hasattr module because torch.nn.DataParallel. Else, just pull model.bert. in single gpu case
                            pooled_output = self.model.module.bert.pooler(hidden_states[-1])
                            pooled_output = self.model.module.dropout(pooled_output)
                        else: #single-gpu
                            pooled_output = self.model.bert.pooler(hidden_states[-1])
                            pooled_output = self.model.dropout(pooled_output)
                        # logits = self.model.classifier(pooled_output)
                    elif args.model_type == "roberta":
                        if args.n_gpu > 1:
                            pooled_output = self.model.module.classifier.dropout(hidden_states[-1][:, 0, :])
                            pooled_output = self.model.module.classifier.dense(pooled_output)
                            pooled_output = torch.tanh(pooled_output)
                            pooled_output = self.model.module.classifier.dropout(pooled_output)
                        # logits = self.model.classifier.out_proj(pooled_output)
                        else:
                            pooled_output = self.model.classifier.dropout(hidden_states[-1][:, 0, :])
                            pooled_output = self.model.classifier.dense(pooled_output)
                            pooled_output = torch.tanh(pooled_output)
                            pooled_output = self.model.classifier.dropout(pooled_output)
                    elif args.model_type == "xlnet":
                        if args.n_gpu > 1:
                            pooled_output = self.model.module.sequence_summary(hidden_states[-1])
                        else:
                            pooled_output = self.model.sequence_summary(hidden_states[-1])

                        # logits = self.model.logits_proj(pooled_output)
                    else:
                        raise NotImplementedError("unknown args.model_type {}".format(args.model_type))
                    all_pooled_output.append(pooled_output.cpu().numpy())

            # get topk prediction rows,cols,vals
            cpred_topk_vals, cpred_topk_cols = c_pred.topk(topk, dim=1)
            cpred_topk_rows = total_example + torch.arange(cur_batch_size)
            cpred_topk_rows = cpred_topk_rows.view(cur_batch_size, 1).expand_as(cpred_topk_cols)
            total_example += cur_batch_size

            # append
            rows += cpred_topk_rows.numpy().flatten().tolist()
            cols += cpred_topk_cols.cpu().numpy().flatten().tolist()
            vals += cpred_topk_vals.cpu().numpy().flatten().tolist()

        eval_loss = total_loss / total_example
        m = int(total_example)
        n = self.num_clusters
        pred_csr_codes = smat.csr_matrix((vals, (rows, cols)), shape=(m, n))
        pred_csr_codes = rf_util.smat_util.sorted_csr(pred_csr_codes, only_topk=args.only_topk)
        C_eval_pred = pred_csr_codes

        # evaluation
        eval_metrics = rf_linear.Metrics.generate(C_eval_true, C_eval_pred, topk=args.only_topk)
        if get_hidden:
            eval_embeddings = np.concatenate(all_pooled_output, axis=0)
        else:
            eval_embeddings = None
        return eval_loss, eval_metrics, C_eval_pred, eval_embeddings

    def train(self, args, X_trn, C_trn):
        """ Train the model """
        args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
        all_inst_idx = torch.tensor([f["inst_idx"] for f in X_trn], dtype=torch.long)
        all_input_ids = torch.tensor([f["input_ids"] for f in X_trn], dtype=torch.long)
        all_attention_mask = torch.tensor([f["attention_mask"] for f in X_trn], dtype=torch.long)
        all_token_type_ids = torch.tensor([f["token_type_ids"] for f in X_trn], dtype=torch.long)
        train_data = TensorDataset(all_inst_idx, all_input_ids, all_attention_mask, all_token_type_ids)
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4,)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
            )

        # Start Batch Training
        if args.local_rank in [-1, 0]:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(X_trn))
            logger.info("  Num Epochs = %d", args.num_train_epochs)
            logger.info("  Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
            logger.info(
                "  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
            )
            logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
            logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        total_run_time = 0.0
        best_matcher_prec = -1

        self.model.zero_grad()
        set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
        for epoch in range(1, int(args.num_train_epochs) + 1):
            for step, batch in enumerate(train_dataloader):
                self.model.train()
                start_time = time.time()
                inst_idx = batch[0]
                inputs = {
                    "input_ids": batch[1].to(args.device),
                    "attention_mask": batch[2].to(args.device),
                }
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[3].to(args.device) if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM, DistilBERT and RoBERTa don't use segment_ids

                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                )
                logits = outputs[0]  # model outputs are always tuple in transformers (see doc)

                # compute loss, average across multi-gpu
                labels = np.array(C_trn[inst_idx].toarray())
                labels = torch.tensor(labels, dtype=torch.float).to(args.device)
                loss = self.loss_fn(logits, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                total_run_time += time.time() - start_time
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # print training log
                        elapsed = time.time() - start_time
                        cur_loss = (tr_loss - logging_loss) / args.logging_steps
                        cur_lr = scheduler.get_lr()[0]
                        logger.info(
                            "| [{:4d}/{:4d}][{:6d}/{:6d}] | {:4d}/{:4d} batches | ms/batch {:5.4f} | train_loss {:6e} | lr {:.6e}".format(
                                int(epoch),
                                int(args.num_train_epochs),
                                int(global_step),
                                int(t_total),
                                int(step),
                                len(train_dataloader),
                                elapsed * 1000.0 / args.logging_steps,
                                cur_loss,
                                cur_lr,
                            )
                        )
                        logging_loss = tr_loss

                if args.max_steps > 0 and global_step > args.max_steps:
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                break


def main():
    # get args
    args = TransformerMatcher.get_args_and_set_logger()["args"]

    # do_train and save model
    if args.do_train:
        # setup output_dir
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir)
            )
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # load data
        with open(args.trn_feat_path, "rb") as fin:
            X_trn = pickle.load(fin)
        C_trn = smat.load_npz(args.trn_label_path)

        # prepare transformer pretrained models
        TransformerMatcher.set_device(args)
        matcher = TransformerMatcher(num_clusters=C_trn.shape[1])
        matcher.prepare_model(args)

        # train
        matcher.train(args, X_trn, C_trn)
        if args.local_rank in [-1, 0]:
            matcher.save_model(args)

    # do_eval on test set and save prediction output
    if args.do_eval:
        # we only support multigpu mode but not distributed mode
        assert args.local_rank == -1

        # load data
        with open(args.trn_feat_path, "rb") as fin:
            X_trn = pickle.load(fin)
        with open(args.tst_feat_path, "rb") as fin:
            X_tst = pickle.load(fin)
        C_trn = smat.load_npz(args.trn_label_path)
        C_tst = smat.load_npz(args.tst_label_path)

        # load fine-tuned model in the args.output_dir
        TransformerMatcher.set_device(args)
        matcher = TransformerMatcher(num_clusters=C_trn.shape[1])
        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        matcher.config = config_class.from_pretrained(args.output_dir)
        matcher.config.output_hidden_states = True
        model = model_class.from_pretrained(args.output_dir, config=matcher.config)
        model.to(args.device)
        matcher.model = model

        # predict
        trn_loss, trn_metrics, C_trn_pred, trn_embeddings = matcher.predict(args, X_trn, C_trn, topk=args.only_topk, get_hidden=True)
        tst_loss, tst_metrics, C_tst_pred, tst_embeddings = matcher.predict(args, X_tst, C_tst, topk=args.only_topk, get_hidden=True)
        logger.info("| matcher_trn_prec {}".format(" ".join("{:4.2f}".format(100 * v) for v in trn_metrics.prec)))
        logger.info("| matcher_trn_recl {}".format(" ".join("{:4.2f}".format(100 * v) for v in trn_metrics.recall)))
        logger.info("| matcher_tst_prec {}".format(" ".join("{:4.2f}".format(100 * v) for v in tst_metrics.prec)))
        logger.info("| matcher_tst_recl {}".format(" ".join("{:4.2f}".format(100 * v) for v in tst_metrics.recall)))

        # save C_trn_pred.npz and trn_embedding.npy
        trn_csr_codes = rf_util.smat_util.sorted_csr(C_trn_pred, only_topk=args.only_topk)
        trn_csr_codes = transform_prediction(trn_csr_codes, transform="lpsvm-l2")
        csr_codes_path = os.path.join(args.output_dir, "C_trn_pred.npz")
        smat.save_npz(csr_codes_path, trn_csr_codes)
        embedding_path = os.path.join(args.output_dir, "trn_embeddings.npy")
        np.save(embedding_path, trn_embeddings)

        # save C_eval_pred.npz and tst_embedding.npy
        tst_csr_codes = rf_util.smat_util.sorted_csr(C_tst_pred, only_topk=args.only_topk)
        tst_csr_codes = transform_prediction(tst_csr_codes, transform="lpsvm-l2")
        csr_codes_path = os.path.join(args.output_dir, "C_tst_pred.npz")
        smat.save_npz(csr_codes_path, tst_csr_codes)
        embedding_path = os.path.join(args.output_dir, "tst_embeddings.npy")
        np.save(embedding_path, tst_embeddings)


if __name__ == "__main__":
    main()
