#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import pickle
import scipy as sp
import scipy.sparse as smat
import xbert.rf_linear as rf_linear


def print_ens(Y_true, Y_pred_list):
    for ens in [
        rf_linear.CsrEnsembler.average,
        rf_linear.CsrEnsembler.rank_average,
        rf_linear.CsrEnsembler.round_robin,
    ]:
        print("ens: {}".format(ens.__name__))
        print(rf_linear.Metrics.generate(Y_true, ens(*Y_pred_list)))


def main(args):
    # loading test set
    Y_true = smat.load_npz(args.input_inst_label)
    Y_pred_list = []
    for pred_path in args.pred_path:
        if not os.path.exists(pred_path):
            raise Warning("pred_path does not exists: {}".format(pred_path))
        else:
            Y_pred = smat.load_npz(pred_path)
            Y_pred.data = rf_linear.Transform.sigmoid(Y_pred.data)

            Y_pred_list += [Y_pred]
            print("==== Evaluation on {}".format(pred_path))
            print(rf_linear.Metrics.generate(Y_true, Y_pred))
    if args.ensemble and len(Y_pred_list) > 1:
        print("==== Evaluations of Ensembles of All Predictions ====")
        for ens in [
            rf_linear.CsrEnsembler.average,
            rf_linear.CsrEnsembler.rank_average,
            rf_linear.CsrEnsembler.round_robin,
        ]:
            print("ens: {}".format(ens.__name__))
            print(rf_linear.Metrics.generate(Y_true, ens(*Y_pred_list)))


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y",
        "--input-inst-label",
        type=str,
        required=True,
        help="path to the npz file of the truth label matrix (CSR) for computing metrics",
    )
    parser.add_argument(
        "-e", "--ensemble", action="store_true", help="whether to perform ensemble evaluations as well",
    )
    parser.add_argument(
        "-p", "--pred_path", nargs="+", help="path to the npz file of the sorted prediction (CSR)",
    )
    args = parser.parse_args()
    print(args)
    main(args)
