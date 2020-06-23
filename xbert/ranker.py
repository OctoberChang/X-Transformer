# coding=utf-8

import argparse
import sys
from os import path
import numpy as np
import scipy as sp
import scipy.sparse as smat
from sklearn.preprocessing import normalize as sk_normalize
from xbert.rf_linear import MLProblem, Metrics, HierarchicalMLModel, PostProcessor, LabelEmbeddingFactory
from xbert.indexer import Indexer

# solver_type
solver_dict = {
    #'L2R_LR':0,
    "L2R_L2LOSS_SVC_DUAL": 1,
    #'L2R_L2LOSS_SVC':2,
    "L2R_L1LOSS_SVC_DUAL": 3,
    #'MCSVM_CS':4,
    "L1R_L2LOSS_SVC": 5,
    #'L1R_LR':6,
    "L2R_LR_DUAL": 7,
}


class LinearModel(object):
    def __init__(self, model=None):
        self.model = model

    def __getitem__(self, key):
        return LinearModel(self.model[key])

    def __add__(self, other):
        return LinearModel(self.model + other.model, self.bias)

    def save(self, model_folder):
        self.model.save(model_folder)

    @classmethod
    def load(cls, model_folder):
        return cls(HierarchicalMLModel.load(model_folder))

    @classmethod
    def train(
        cls,
        X,
        Y,
        C,
        mode="full-model",
        shallow=False,
        solver_type=solver_dict["L2R_L2LOSS_SVC_DUAL"],
        Cp=1.0,
        Cn=1.0,
        threshold=0.1,
        max_iter=100,
        threads=-1,
        bias=-1.0,
        Z_pred=None,
        negative_sampling_scheme=None,
    ):
        if mode in ["full-model", "matcher"]:
            if mode == "full-model":
                prob = MLProblem(X, Y, C, Z_pred=Z_pred, negative_sampling_scheme=negative_sampling_scheme,)
            elif mode == "matcher":
                assert C is not None
                Y = Y.dot(C)
                prob = MLProblem(X, Y, C=None)

            hierarchical = True
            min_labels = 2
            if shallow:
                if prob.C is None:
                    min_labels = prob.Y.shape[1]
                else:
                    min_labels = prob.C.shape[1]
        elif mode == "ranker":
            assert C is not None
            prob = MLProblem(X, Y, C, Z_pred=Z_pred, negative_sampling_scheme=negative_sampling_scheme,)
            hierarchical = False
            min_labels = 2

        model = HierarchicalMLModel.train(
            prob,
            hierarchical=hierarchical,
            min_labels=min_labels,
            solver_type=solver_type,
            Cp=Cp,
            Cn=Cn,
            threshold=threshold,
            threads=threads,
            bias=bias,
            max_iter=max_iter,
        )
        return cls(model)

    def predict(self, X, csr_codes=None, beam_size=10, only_topk=10, cond_prob=True):
        pred_csr = self.model.predict(X, only_topk=only_topk, csr_codes=csr_codes, beam_size=beam_size, cond_prob=cond_prob,)
        return pred_csr


class SubCommand(object):
    def __init__(self):
        pass

    @classmethod
    def add_parser(cls, super_parser):
        pass

    @staticmethod
    def add_arguments(parser):
        pass


def load_feature_matrix(args):
    if args.feature_format % 3 == 0:
        X1 = HierarchicalMLModel.load_feature_matrix(args.input_inst_feat1)
        X2 = HierarchicalMLModel.load_feature_matrix(args.input_inst_feat2)
        X = smat.hstack([sk_normalize(X1, axis=1), sk_normalize(X2, axis=1)]).tocsr()
    elif args.feature_format % 3 == 1 and args.input_inst_feat1:
        X = HierarchicalMLModel.load_feature_matrix(args.input_inst_feat1)
    elif args.feature_format % 3 == 2 and args.input_inst_feat2:
        X = HierarchicalMLModel.load_feature_matrix(args.input_inst_feat2)
    else:
        raise NotImplementedError(f"args.feature_format = {args.feature_format} is not supported.")
    if args.feature_format // 3 == 0:
        X = sk_normalize(X, axis=1, copy=False)
    return X

class LinearTrainCommand(SubCommand):
    @staticmethod
    def run(args):
        X = load_feature_matrix(args)
        Y = smat.load_npz(args.input_inst_label)
        label_feat = LabelEmbeddingFactory.create(Y, X, method=args.input_label_feat, dtype=X.dtype)
        C = Indexer.load_indexed_code(args.input_code_path, label_feat)
        if args.pred_inst_codes is not None:
            Z_pred = smat.load_npz(args.pred_inst_codes)
        else:
            Z_pred = None
        model = LinearModel.train(
            X,
            Y,
            C,
            mode=args.mode,
            shallow=args.shallow,
            solver_type=solver_dict[args.solver_type],
            Cp=args.Cp,
            Cn=args.Cn,
            threshold=args.threshold,
            threads=args.threads,
            bias=args.bias,
            Z_pred=Z_pred,
            negative_sampling_scheme=args.negative_sampling_scheme,
        )
        model.save(args.output_ranker_folder)

    @classmethod
    def add_parser(cls, super_parser):
        parser = super_parser.add_parser("train", aliases=[], help="Train a linear ranker with codes")
        cls.add_arguments(parser)
        parser.set_defaults(run=cls.run)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "-x",
            "-x1",
            "--input-inst-feat1",
            metavar="PATH",
            type=str,
            required=True,
            help="path to the npz file of the feature matrix (CSR)",
        )

        parser.add_argument(
            "-x2",
            "--input-inst-feat2",
            type=str,
            default=None,
            metavar="PATH",
            help="path to the npz file of the feature matrix (CSR, nr_insts * nr_feats)",
        )

        parser.add_argument(
            "-f",
            "--feature-format",
            dest="feature_format",
            type=int,
            default=1,
            metavar="INT",
            help="feature format: 0=> normalized [x1, x2], 1=> normalized x1, 2=> normalized x2; 3=> [x1 x2], 4=> x1, 5=> x2\n",
        )

        parser.add_argument(
            "-y",
            "--input-inst-label",
            type=str,
            required=True,
            metavar="PATH",
            help="path to the npz file of the label matrix (CSR, nr_insts * nr_labels)",
        )

        parser.add_argument(
            "-z",
            "--pred-inst-codes",
            type=str,
            metavar="PATH",
            help="path to the npz file of the predicted inst-cluster matrix (CSR, nr_insts * nr_codes)",
        )

        parser.add_argument(
            "-c",
            "--input-code-path",
            type=str,
            required=True,
            metavar="PATH",
            help="path to the npz file of the indexing codes (CSR, nr_labels * nr_codes)",
        )

        parser.add_argument(
            "-L",
            "--input-label-feat",
            type=str,
            default=None,
            metavar="PATH",
            help="path to the npz file of the feature matrix (CSR)",
        )

        parser.add_argument(
            "-ns",
            "--negative-sampling-scheme",
            type=int,
            default=1,  # for backward compatibiilty
            metavar="INT",
            help="0: negative from both indexer and matcher, 1: negative from indexer,  2: negative from matcher",
        )

        parser.add_argument(
            "-m",
            "-o",  # for backward compatibility
            "--output-ranker-folder",
            type=str,
            required=True,
            metavar="DIR",
            help="directory for storing linear ranker",
        )

        parser.add_argument(
            "--mode", type=str, default="full-model", metavar="STR", help="mode: [full-model|ranker] (default full-model)",
        )

        parser.add_argument(
            "-S", "--shallow", action="store_true", help="perform shallow linear modeling instead of hierarchical linear modeling",
        )

        parser.add_argument(
            "-s",
            "--solver-type",
            type=str,
            default="L2R_L2LOSS_SVC_DUAL",
            metavar="SOLVER_STR",
            help="{} (default L2R_L2LOSS_SVC_DUAL)".format(" | ".join(solver_dict.keys())),
        )

        parser.add_argument(
            "--Cp", type=float, default=1.0, metavar="VAL", help="coefficient for positive class in the loss function (default 1.0)",
        )

        parser.add_argument(
            "--Cn", type=float, default=1.0, metavar="VAL", help="coefficient for negative class in the loss function (default 1.0)",
        )

        parser.add_argument(
            "-B",
            "--bias",
            type=float,
            default=1.0,
            metavar="bias",
            help="if bias > 0, instance x becomes [x; bias]; if <= 0, no bias term added (default 1.0)",
        )

        parser.add_argument(
            "-t", "--threshold", type=float, default=0.1, metavar="VAL", help="threshold to sparsity the model weights (default 0.1)",
        )

        parser.add_argument(
            "-n", "--threads", type=int, default=-1, metavar="INT", help="number of threads to use (default -1 to denote all the CPUs)",
        )


class LinearPredictCommand(SubCommand):
    @staticmethod
    def run(args):
        Xt = load_feature_matrix(args)
        model = LinearModel.load(args.input_ranker_folder)
        # get only ranker part if predicted_csr_code from a matcher is provided
        if args.predicted_csr_code is not None and path.exists(args.predicted_csr_code):
            csr_codes = smat.load_npz(args.predicted_csr_code)
            model = model[-1]
        else:
            csr_codes = None

        cond_prob = PostProcessor.get(args.transform)
        Yt_pred = model.predict(Xt, csr_codes=csr_codes, beam_size=args.beam_size, only_topk=args.only_topk, cond_prob=cond_prob,)
        if args.input_inst_label is not None and path.exists(args.input_inst_label):
            Yt = smat.load_npz(args.input_inst_label) if args.input_inst_label else None
            metric = Metrics.generate(Yt, Yt_pred, topk=10)
            print("==== tst_set evaluation ====")
            print(metric)

        smat.save_npz(args.output_path, Yt_pred)

    @classmethod
    def add_parser(cls, super_parser):
        parser = super_parser.add_parser("predict", aliases=[], help="Generate predictions based on the given ranker")
        cls.add_arguments(parser)
        parser.set_defaults(run=cls.run)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "-m", "--input-ranker-folder", type=str, required=True, help="path to the ranker folder",
        )

        parser.add_argument(
            "-x", "-x1", "--input-inst-feat1", type=str, required=True, help="path to the npz file of the feature matrix (CSR)",
        )
        parser.add_argument(
            "-x2",
            "--input-inst-feat2",
            type=str,
            default=None,
            metavar="PATH",
            help="path to the npz file of the feature matrix (CSR, nr_insts * nr_feats)",
        )

        parser.add_argument(
            "-f",
            "--feature-format",
            dest="feature_format",
            type=int,
            default=1,
            metavar="INT",
            help="feature format: 0=> normalized [x1, x2], 1=> normalized x1, 2=> normalized x2; 3=> [x1 x2], 4=> x1, 5=> x2\n",
        )

        parser.add_argument(
            "-y",
            "--input-inst-label",
            type=str,
            required=False,
            help="path to the npz file of the label matrix (CSR) for computing metrics",
        )

        parser.add_argument(
            "-o", "--output-path", type=str, required=True, help="path to the npz file of output prediction (CSR)",
        )

        parser.add_argument(
            "-c",
            "-z",
            "--predicted-csr-code",
            type=str,
            required=False,
            help="path to the npz file of the csr codes generated by the matcher",
        )

        parser.add_argument(
            "-t",
            "--transform",
            type=str,
            default="l3-hinge",
            help="transform of the ranker prediction to be multiplied by the input csr codes sigmoid | l1-hinge | l2-hinge | l3-hinge (default l2-hinge)",
        )

        parser.add_argument(
            "-k", "--only-topk", type=int, default=10, help="number of top labels in the prediction",
        )

        parser.add_argument(
            "-b", "--beam-size", type=int, default=10, help="size of beam search in the prediction",
        )


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="subcommands", metavar="SUBCOMMAND")
    subparsers.required = True
    LinearTrainCommand.add_parser(subparsers)
    LinearPredictCommand.add_parser(subparsers)
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    args.run(args)
