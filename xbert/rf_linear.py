import sys
import os
from os import path, system
import time
import collections
import itertools
import pickle
import json
import glob
from glob import glob
import ctypes
from ctypes import *

import scipy as sp
import scipy.sparse as smat
from sklearn.preprocessing import normalize as sk_normalize

import xbert.indexer as indexer
from xbert.rf_util import (
    PyMatrix,
    fillprototype,
    load_dynamic_library,
    COOAllocator,
    PredAllocator,
    smat_util,
)


# solver_type
L2R_LR = 0
L2R_L2LOSS_SVC_DUAL = 1
L2R_L2LOSS_SVC = 2
L2R_L1LOSS_SVC_DUAL = 3
MCSVM_CS = 4
L1R_L2LOSS_SVC = 5
L1R_LR = 6
L2R_LR_DUAL = 7


class corelib(object):
    def __init__(self, dirname, soname, forced_rebuild=False):
        self.clib_float32 = load_dynamic_library(dirname, soname + "_float32", forced_rebuild=forced_rebuild)
        self.clib_float64 = load_dynamic_library(dirname, soname + "_float64", forced_rebuild=forced_rebuild)
        arg_list = [
            POINTER(PyMatrix),  # PyMatrix X
            POINTER(PyMatrix),  # PyMatrix Y
            POINTER(PyMatrix),  # PyMatrix C
            POINTER(PyMatrix),  # PyMatrix Z
            COOAllocator.CFUNCTYPE,  # py_coo_allocator
            c_double,  # threshold
            c_int,  # solver_type
            c_double,  # Cp
            c_double,  # Cn
            c_uint64,  # max_iter
            c_double,  # eps
            c_double,  # bias
            c_int,  # threads
        ]
        fillprototype(self.clib_float32.c_multilabel_train_with_codes, None, arg_list)
        fillprototype(self.clib_float64.c_multilabel_train_with_codes, None, arg_list)

        arg_list = [
            POINTER(PyMatrix),
            POINTER(PyMatrix),
            POINTER(PyMatrix),
            POINTER(PyMatrix),
            PredAllocator.CFUNCTYPE,
            c_int,
        ]
        fillprototype(self.clib_float32.c_multilabel_predict_with_codes, None, arg_list)
        fillprototype(self.clib_float64.c_multilabel_predict_with_codes, None, arg_list)

        arg_list = [
            POINTER(PyMatrix),
            POINTER(PyMatrix),
            c_uint64,
            POINTER(c_uint32),
            POINTER(c_uint32),
            c_void_p,
            c_int,
        ]
        fillprototype(self.clib_float32.c_sparse_inner_products, None, arg_list)
        fillprototype(self.clib_float64.c_sparse_inner_products, None, arg_list)

    def sparse_inner_products(self, pX, pM, X_row_idx, M_col_idx, pred_values=None, threads=-1, verbose=0):
        clib = self.clib_float32
        if pX.dtype == sp.float64:
            clib = self.clib_float64
            assert pM.dtype == sp.float64
            if verbose != 0:
                print("perform float64 computation")
        else:
            clib = self.clib_float32
            assert pM.dtype == sp.float32
            if verbose != 0:
                print("perform float32 computation")

        nnz = len(X_row_idx)
        if pred_values is None or pred_values.dtype != pM.dtype or len(pred_values) != nnz:
            pred_values = sp.zeros(nnz, dtype=pM.dtype)
        clib.c_sparse_inner_products(
            byref(pX),
            byref(pM),
            nnz,
            X_row_idx.ctypes.data_as(POINTER(c_uint32)),
            M_col_idx.ctypes.data_as(POINTER(c_uint32)),
            pred_values.ctypes.data_as(c_void_p),
            threads,
        )
        return pred_values

    def multilabel_predict_with_codes(self, pX, pW, pC, pZ, threads=-1, verbose=0):
        clib = self.clib_float32
        if pX.dtype == sp.float64:
            clib = self.clib_float64
            if verbose != 0:
                print("perform float64 computation")
        else:
            clib = self.clib_float32
            if verbose != 0:
                print("perform float32 computation")
        pred_alloc = PredAllocator(dtype=pX.dtype)
        clib.c_multilabel_predict_with_codes(byref(pX), byref(pW), byref(pC), byref(pZ), pred_alloc.cfunc, threads)
        return pred_alloc.get_pred()

    def multilabel_train_with_codes(
        self,
        pX,
        pY,
        pC,
        pZ,
        threshold=0,
        solver_type=L2R_L2LOSS_SVC_DUAL,
        Cp=1.0,
        Cn=1.0,
        max_iter=1000,
        eps=0.1,
        bias=1.0,
        threads=-1,
        verbose=0,
    ):
        clib = self.clib_float32
        if pX.dtype == sp.float64:
            clib = self.clib_float64
            if verbose != 0:
                print("perform float64 computation")
        else:
            clib = self.clib_float32
            if verbose != 0:
                print("perform float32 computation")
        coo_alloc = COOAllocator(dtype=pX.dtype)
        clib.c_multilabel_train_with_codes(
            byref(pX),
            byref(pY),
            byref(pC) if pC is not None else None,
            byref(pZ) if pZ is not None else None,
            coo_alloc.cfunc,
            threshold,
            solver_type,
            Cp,
            Cn,
            max_iter,
            eps,
            bias,
            threads,
        )
        return coo_alloc.tocsc()


forced_rebuild = False
corelib_path = path.join(path.dirname(path.abspath(__file__)), "corelib/")
soname = "rf_linear"
clib = corelib(corelib_path, soname, forced_rebuild)


class WallTimer(object):
    def __init__(self):
        self.last_time = 0

    def now(self):
        return time.time()

    def tic(self):
        self.last_time = self.now()

    def toc(self):
        return (self.now() - self.last_time) * 1e3


class Metrics(collections.namedtuple("Metrics", ["prec", "recall"])):
    __slots__ = ()

    def __str__(self):
        fmt = lambda key: " ".join("{:4.2f}".format(100 * v) for v in getattr(self, key)[:])
        return "\n".join("{:7}= {}".format(key, fmt(key)) for key in self._fields)

    @classmethod
    def default(cls):
        return cls(prec=[], recall=[])

    @classmethod
    def generate(cls, tY, pY, topk=10):
        assert isinstance(tY, smat.csr_matrix), type(tY)
        assert isinstance(pY, smat.csr_matrix), type(pY)
        assert tY.shape == pY.shape, "tY.shape = {}, pY.shape = {}".format(tY.shape, pY.shape)
        pY = smat_util.sorted_csr(pY)
        total_matched = sp.zeros(topk, dtype=sp.uint64)
        recall = sp.zeros(topk, dtype=sp.float64)
        for i in range(tY.shape[0]):
            truth = tY.indices[tY.indptr[i] : tY.indptr[i + 1]]
            matched = sp.isin(pY.indices[pY.indptr[i] : pY.indptr[i + 1]][:topk], truth)
            cum_matched = sp.cumsum(matched, dtype=sp.uint64)
            total_matched[: len(cum_matched)] += cum_matched
            recall[: len(cum_matched)] += cum_matched / len(truth)
            if len(cum_matched) != 0:
                total_matched[len(cum_matched) :] += cum_matched[-1]
                recall[len(cum_matched) :] += cum_matched[-1] / len(truth)
        prec = total_matched / tY.shape[0] / sp.arange(1, topk + 1)
        recall = recall / tY.shape[0]
        return cls(prec=prec, recall=recall)


class Transform(object):
    @staticmethod
    def identity(v, inplace=False):
        return v

    @staticmethod
    def log_lpsvm(p, v, inplace=False):
        if inplace:
            out = v
        else:
            out = sp.zeros_like(v)
        out[:] = -(sp.maximum(1.0 - v, 0) ** p)
        return out

    @staticmethod
    def lpsvm(p, v, inplace=False):
        out = Transform.log_lpsvm(p, v, inplace)
        sp.exp(out, out=out)
        return out

    @staticmethod
    def get_log_lpsvm(p):
        def f(v, inplace=False):
            return Transform.log_lpsvm(p, v, inplace)

        return f

    @staticmethod
    def get_lpsvm(p):
        def f(v, inplace=False):
            return Transform.lpsvm(p, v, inplace)

        return f

    @staticmethod
    def sigmoid(v, inplace=False):
        if inplace:
            out = v
        else:
            out = sp.zeros_like(v)
        out[:] = 1.0 / (1.0 + sp.exp(-v))
        return out

    @staticmethod
    def log_sigmoid(v, inplace=False):
        out = Transform.sigmoid(v, inplace)
        out[:] = sp.log(out)
        return out


class Combiner(object):
    @staticmethod
    def noop(x, y):
        return x

    @staticmethod
    def add(x, y):
        x[:] += y[:]
        return x

    @staticmethod
    def mul(x, y):
        x[:] *= y[:]
        return x

    @staticmethod
    def max(x, y):
        x[:] = sp.maximum(x[:], y[:])
        return x

    @staticmethod
    def noisyor(x, y):
        x[:] = 1.0 - (1.0 - x[:]) * (1.0 - y[:])
        return x


class PostProcessor(object):
    def __init__(self, transform, combiner):
        self.transform = transform
        self.combiner = combiner

    @classmethod
    def get(cls, name):
        mapping = {
            "sigmoid": PostProcessor.sigmoid(),
            "log-sigmoid": cls(Transform.log_sigmoid, Combiner.add),
            "noop": cls(Transform.identity, Combiner.noop),
        }
        for p in [1, 2, 3, 4, 5, 6]:
            mapping["l{}-hinge".format(p)] = cls(Transform.get_lpsvm(p), Combiner.mul)
            mapping["log-l{}-hinge".format(p)] = cls(Transform.get_log_lpsvm(p), Combiner.add)
            mapping["l{}-hinge-noisyor".format(p)] = cls(Transform.get_lpsvm(p), Combiner.noisyor)
        return mapping[name]

    @classmethod
    def sigmoid(cls):
        return cls(Transform.sigmoid, Combiner.mul)

    @classmethod
    def l2svm(cls):
        return cls(Transform.l2svm, Combiner.mul)

    @classmethod
    def noisyor_l2svm(cls):
        return cls(Transform.l2svm, Combiner.noisyor)

    @classmethod
    def noisyor_sigmoid(cls):
        return cls(Transform.sigmoid, Combiner.noisyor)

class LabelEmbeddingFactory(object):

    @staticmethod
    def create(Y, X, method="pifa", dtype=sp.float32):
        mapping = {
            "pifa": LabelEmbeddingFactory.pifa,
            "homer": LabelEmbeddingFactory.homer,
            "spectral": LabelEmbeddingFactory.spectral,
            "none": lambda Y, X, dtype: None,
        }
        if method is None:
            method = "none"
        if method.lower() in mapping:
            return mapping[method.lower()](Y, X, dtype)
        elif (method.endswith(".npz") or method.endswith(".npy")) and path.exists(method):
            label_embedding = HierarchicalMLModel.load_feature_matrix(method, dtype=dtype)
            assert label_embedding.shape[0] == Y.shape[1], f"{label_embedding.shape[0]} != Y.{shape[1]}"
            return label_embedding
        else:
            assert False, f"Something wrong with this label embedding '{method}'. valid ones {mapping.keys()}"

    @staticmethod
    def pifa(Y, X, dtype=sp.float32):
        Y_avg = sk_normalize(Y, axis=1, norm="l2")
        label_embedding = smat.csr_matrix(Y_avg.T.dot(X), dtype=dtype)
        return label_embedding

    @staticmethod
    def homer(Y, X, dtype=sp.float32):
        label_embedding = smat.csr_matrix(Y.T, dtype=dtype)
        return label_embedding

    @staticmethod
    def spectral(Y, X, dtype=sp.float32):
        from sklearn.cluster import SpectralCoclustering
        def scale_normalize(X):
            " from https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/cluster/_bicluster.py#L108"
            row_diag = sp.asarray(sp.sqrt(X.sum(axis=1))).squeeze()
            col_diag = sp.asarray(sp.sqrt(X.sum(axis=0))).squeeze()
            row_diag[row_diag == 0] = 1.0;
            col_diag[col_diag == 0] = 1.0;
            row_diag= 1.0 / row_diag
            col_diag= 1.0 / col_diag
            if smat.issparse(X):
                n_rows, n_cols = X.shape
                r = smat.dia_matrix((row_diag, [0]), shape=(n_rows, n_rows))
                c = smat.dia_matrix((col_diag, [0]), shape=(n_cols, n_cols))
                an = r * X * c
            else:
                an = row_diag[:, sp.newaxis] * X * col_diag
            return an, row_diag, col_diag

        coclustering = SpectralCoclustering(n_clusters=16384, random_state=1)
        normalized_data, row_diag, col_diag = scale_normalize(Y.T)
        n_sv = 1 + int(sp.ceil(sp.log2(coclustering.n_clusters)))
        u, v = coclustering._svd(normalized_data, n_sv, n_discard=1)
        label_embedding = smat.csr_matrix(u, dtype=dtype)
        return label_embedding

class MLProblem(object):
    def __init__(self, X, Y, C=None, dtype=None, Z_pred=None, negative_sampling_scheme=None):
        if dtype is None:
            dtype = X.dtype
        self.pX = PyMatrix.init_from(X, dtype)
        self.pY = PyMatrix.init_from(Y, dtype)
        self.pC = PyMatrix.init_from(C, dtype)
        Z = None if C is None else smat.csr_matrix(self.Y.dot(self.C))
        if negative_sampling_scheme is None or negative_sampling_scheme == 1:
            Z = Z
        elif negative_sampling_scheme is not None:
            if negative_sampling_scheme == 0:
                Z = (Z + Z_pred).tocsr()
            elif negative_sampling_scheme == 1:
                Z = Z
            elif negative_sampling_scheme == 2 and Z_pred is not None:
                Z = Z_pred
        self.pZ = PyMatrix.init_from(Z, dtype)  # Z = Y * C
        self.dtype = dtype

    @property
    def X(self):
        return None if self.pX is None else self.pX.buf

    @property
    def Y(self):
        return None if self.pY is None else self.pY.buf

    @property
    def C(self):
        return None if self.pC is None else self.pC.buf

    @property
    def Z(self):
        return None if self.pZ is None else self.pZ.buf

    @property
    def nr_labels(self):
        return None if self.pY is None else self.Y.shape[1]


class MLModel(object):
    def __init__(self, W, C=None, dtype=None):
        if C is not None:
            if isinstance(C, PyMatrix):
                assert C.buf.shape[0] == W.shape[1]
            else:
                assert C.shape[0] == W.shape[1], "C:{} W:{}".format(C.shape, W.shape)
        if dtype is None:
            dtype = W.dtype
        self.pC = PyMatrix.init_from(C, dtype)
        self.pW = PyMatrix.init_from(W, dtype)

    @property
    def C(self):
        return None if self.pC is None else self.pC.buf

    @property
    def W(self):
        return None if self.pW is None else self.pW.buf

    @property
    def nr_labels(self):
        return self.W.shape[1]

    @property
    def nr_codes(self):
        return 0 if self.C is None else self.C.shape[1]

    @property
    def nr_features(self):
        return self.W.shape[0]

    @property
    def dtype(self):
        return self.pW.dtype

    def astype(self, dtype):
        if dtype == self.pW.dtype:
            return self
        else:
            return MLModel(self.W, self.C, dtype)

    @classmethod
    def load(cls, folder, dtype=None):
        param = json.loads(open("{}/param.json".format(folder), "r").read())
        assert param["model"] == cls.__name__
        W = smat.load_npz("{}/W.npz".format(folder)).sorted_indices()
        if path.exists("{}/C.npz".format(folder)):
            C = smat.load_npz("{}/C.npz".format(folder)).sorted_indices()
        else:
            C = None
        return cls(W, C, dtype=dtype)

    def save(self, folder):
        if not path.exists(folder):
            os.makedirs(folder)
        param = {
            "model": self.__class__.__name__,
            "nr_labels": self.nr_labels,
            "nr_features": self.nr_features,
            "nr_codes": self.nr_codes,
        }
        open("{}/param.json".format(folder), "w").write(json.dumps(param, indent=True))
        smat.save_npz("{}/W.npz".format(folder), self.W, compressed=False)
        if self.C is not None:
            smat.save_npz("{}/C.npz".format(folder), self.C, compressed=False)

    @classmethod
    def train(
        cls,
        prob,
        threshold=0.0,
        solver_type=L2R_L2LOSS_SVC_DUAL,
        Cp=1.0,
        Cn=1.0,
        max_iter=100,
        eps=0.1,
        bias=1.0,
        threads=-1,
        verbose=0,
        **arg_kw,
    ):
        model = clib.multilabel_train_with_codes(
            prob.pX,
            prob.pY,
            prob.pC,
            prob.pZ,
            threshold=threshold,
            solver_type=solver_type,
            Cp=Cp,
            Cn=Cn,
            max_iter=max_iter,
            eps=eps,
            bias=bias,
            threads=threads,
            verbose=verbose,
        )
        return cls(model, prob.pC)

    def predict(
        self, X, only_topk=None, csr_codes=None, cond_prob=None, normalized=False, threads=-1,
    ):
        assert X.shape[1] == self.nr_features
        if csr_codes is None:
            dense = X.dot(self.W).toarray()
            if cond_prob:
                dense = cond_prob.transform(dense, inplace=True)
            coo = smat_util.dense_to_coo(dense)
            pred_csr = smat_util.sorted_csr_from_coo(coo.shape, coo.row, coo.col, coo.data, only_topk=only_topk)
        else:  # csr_codes is given
            assert self.C is not None, "This model does not have C"
            assert X.shape[1] == self.nr_features
            assert csr_codes.shape[0] == X.shape[0]
            assert csr_codes.shape[1] == self.nr_codes
            if (csr_codes.data == 0).sum() != 0:
                # this is a trick to avoid zero entries explicit removal from the smat_dot_smat
                offset = sp.absolute(csr_codes.data).max() + 1
                csr_codes = smat.csr_matrix((csr_codes.data + offset, csr_codes.indices, csr_codes.indptr), shape=csr_codes.shape,)
                csr_labels = (csr_codes.dot(self.C.T)).tocsr()
                csr_labels.data -= offset
            else:
                csr_labels = (csr_codes.dot(self.C.T)).tocsr()
            nnz_of_insts = csr_labels.indptr[1:] - csr_labels.indptr[:-1]
            inst_idx = sp.repeat(sp.arange(X.shape[0], dtype=sp.uint32), nnz_of_insts)
            label_idx = csr_labels.indices.astype(sp.uint32)
            val = self.predict_values(X, inst_idx, label_idx, threads=threads)
            if cond_prob:
                val = cond_prob.transform(val, inplace=True)
                val = cond_prob.combiner(val, csr_labels.data)

            pred_csr = smat_util.sorted_csr_from_coo(csr_labels.shape, inst_idx, label_idx, val, only_topk=only_topk)

        if normalized:
            pred_csr = sk_normalize(pred_csr, axis=1, copy=False, norm="l1")
        return pred_csr

    def predict_new(
        self, X, only_topk=None, csr_codes=None, cond_prob=None, normalized=False, threads=-1,
    ):
        assert X.shape[1] == self.nr_features
        if csr_codes is None:
            dense = X.dot(self.W).toarray()
            if cond_prob:
                dense = cond_prob.transform(dense, inplace=True)
            coo = smat_util.dense_to_coo(dense)
            pred_csr = smat_util.sorted_csr_from_coo(coo.shape, coo.row, coo.col, coo.data, only_topk=only_topk)
        else:  # csr_codes is given
            assert self.C is not None, "This model does not have C"
            assert X.shape[1] == self.nr_features
            assert csr_codes.shape[0] == X.shape[0]
            assert csr_codes.shape[1] == self.nr_codes
            if not csr_codes.has_sorted_indices:
                csr_codes = csr_codes.sorted_indices()
            if (csr_codes.data == 0).sum() != 0:
                # this is a trick to avoid zero entries explicit removal from the smat_dot_smat
                offset = sp.absolute(csr_codes.data).max() + 1
                csr_codes = smat.csr_matrix((csr_codes.data + offset, csr_codes.indices, csr_codes.indptr), shape=csr_codes.shape,)
                pZ = PyMatrix.init_from(csr_codes, self.dtype)
                csr_labels, pred_csr = clib.multilabel_predict_with_codes(X, self.pW, self.pC, pZ, threads=threads)
                csr_labels.data -= offset
            else:
                pZ = PyMatrix.init_from(csr_codes.sorted_indices(), self.dtype)
                csr_labels, pred_csr = clib.multilabel_predict_with_codes(X, self.pW, self.pC, pZ, threads=threads)
            val = pred_csr.data
            if cond_prob:
                val = cond_prob.transform(val, inplace=True)
                val = cond_prob.combiner(val, csr_labels.data)

            pred_csr = smat_util.sorted_csr(pred_csr, only_topk=only_topk)

        if normalized:
            pred_csr = sk_normalize(pred_csr, axis=1, copy=False, norm="l1")
        return pred_csr

    def predict_values(self, X, inst_idx, label_idx, out=None, threads=-1):
        assert X.shape[1] == self.nr_features
        if out is None:
            out = sp.zeros(inst_idx.shape, dtype=self.pW.dtype)
        pX = PyMatrix.init_from(X, dtype=self.pW.dtype)
        out = clib.sparse_inner_products(pX, self.pW, inst_idx.astype(sp.uint32), label_idx.astype(sp.uint32), out, threads=threads,)
        return out

    def predict_with_coo_labels(self, X, inst_idx, label_idx, only_topk=None):
        val = self.predict_values(X, inst_idx, label_idx)
        shape = (X.shape[0], self.nr_labels)
        pred_csr = smat_util.sorted_csr_from_coo(shape, inst_idx, label_idx, val, only_topk=only_topk)
        return pred_csr

    def predict_with_csr_labels(self, X, csr_labels, only_topk=None):
        assert X.shape[1] == self.nr_features
        assert csr_labels.shape[0] == X.shape[0]
        assert csr_labels.shape[1] == self.nr_labels
        nz_of_rows = csr_labels.indptr[1:] - csr_labels.indptr[:-1]
        inst_idx = sp.repeat(sp.arange(X.shape[0]), nz_of_rows).astype(sp.uint32)
        label_idx = csr_labels.indices
        return self.predict_with_coo_labels(X, inst_idx, label_idx, only_topk)

    def predict_with_coo_codes(self, X, inst_idx, code_idx, only_topk=None):
        assert self.C != None, "This Model does not have codes"
        shape = (X.shape[0], self.nr_codes)
        tmp_ones = sp.ones_like(code_idx)
        csr_codes = smat.csr_matrix((tmp_ones, (inst_idx, code_idx)), shape=shape, dtype=sp.float32)
        coo_labels = (csr_codes.dot(self.C.T)).tocoo()
        return self.predict_with_coo_labels(X, coo_labels.row, coo_labels.col, only_topk)

    def predict_with_csr_codes(self, X, csr_codes, only_topk=None):
        assert self.C != None, "This Model does not have codes"
        assert X.shape[1] == self.nr_features
        assert csr_codes.shape[0] == X.shape[0]
        assert csr_codes.shape[1] == self.nr_codes
        coo_labels = (csr_codes.dot(self.C.T)).tocoo()
        return self.predict_with_coo_labels(X, coo_labels.row, coo_labels.col, only_topk)


class HierarchicalMLModel(object):
    """A hierachical linear multilable model"""

    def __init__(self, model_chain, bias=-1):
        if isinstance(model_chain, (list, tuple)):
            self.model_chain = model_chain
        else:
            self.model_chain = [model_chain]
        self.bias = bias

    @staticmethod
    def load_feature_matrix(src, dtype=sp.float32):
        if src.endswith(".npz"):
            return smat.load_npz(src).tocsr().astype(dtype)
        elif src.endswith(".npy"):
            return smat.csr_matrix(sp.ascontiguousarray(sp.load(src), dtype=dtype))
        else:
            raise ValueError("src must end with .npz or .npy")

    @property
    def depth(self):
        return len(self.model_chain)

    @property
    def nr_features(self):
        return self.model_chain[0].nr_features - (1 if self.bias > 0 else 0)

    @property
    def nr_codes(self):
        return self.model_chain[-1].nr_codes

    @property
    def nr_labels(self):
        return self.model_chain[-1].nr_labels

    def __add__(self, other):
        if not isinstance(other, HierarchicalMLModel):
            other = HierarchicalMLModel(other)
        assert self.model_chain[-1].nr_labels == other.model_chain[0].nr_codes
        return HierarchicalMLModel(self.model_chain + other.model_chain, self.bias)

    def __getitem__(self, key):
        return HierarchicalMLModel(self.model_chain[key], self.bias)

    def astype(self, dtype):
        if dtype == self.model_chain[0].dtype:
            return self
        else:
            return HierarchicalMLModel([m.astype(dtype) for m in self.model_chain])

    @classmethod
    def load(cls, folder, dtype=None):
        param = json.loads(open("{}/param.json".format(folder), "r").read())
        assert param["model"] == cls.__name__
        depth = int(param.get("depth", len(glob("{}/*.model".format(folder)))))

        bias = float(param.get("bias", -1.0))  # backward compatibility in case bias term is not listed in param.json
        return cls([load_model("{}/{}.model".format(folder, d), dtype=dtype) for d in range(depth)], bias,)

    def save(self, folder):
        if not path.exists(folder):
            os.makedirs(folder)
        depth = self.depth
        param = {
            "model": self.__class__.__name__,
            "depth": self.depth,
            "nr_features": self.nr_features,
            "nr_codes": self.nr_codes,
            "nr_labels": self.nr_labels,
            "bias": self.bias,
        }
        open("{}/param.json".format(folder), "w").write(json.dumps(param, indent=True))
        for d in range(depth):
            local_folder = "{}/{}.model".format(folder, d)
            self.model_chain[d].save(local_folder)

    @classmethod
    def train(cls, prob, hierarchical=None, min_labels=2, nr_splits=2, **arg_kw):
        if hierarchical is None or hierarchical == False:
            return HierarchicalMLModel([MLModel.train(prob, **arg_kw)], arg_kw.get("bias", 1.0))

        model_chain = []
        cur_prob = prob
        if min_labels <= 1:
            min_labels = prob.C.shape[1]
        while True:
            if cur_prob.C is None and cur_prob.nr_labels > min_labels:
                cur_codes = sp.arange(cur_prob.nr_labels)
                new_codes = cur_codes // nr_splits
                shape = (len(cur_codes), new_codes.max() + 1)
                newC = smat.csr_matrix((sp.ones_like(cur_codes), (cur_codes, new_codes)), shape=shape)
                cur_prob = MLProblem(cur_prob.pX, cur_prob.pY, newC)
            cur_model = MLModel.train(cur_prob, **arg_kw)
            model_chain += [cur_model]
            if cur_model.C is None:
                break
            else:
                newY = cur_prob.Y.dot(cur_prob.C)
                cur_prob = MLProblem(cur_prob.pX, newY)
        model_chain = model_chain[::-1]
        return cls(model_chain, arg_kw.get("bias", 1.0))

    def predict(
        self, X, only_topk=None, csr_codes=None, beam_size=2, max_depth=None, cond_prob=True, normalized=False, threads=-1,
    ):
        if max_depth is None:
            max_depth = self.depth
        if cond_prob is None or cond_prob == False:
            cond_prob = PostProcessor(Transform.identity, Combiner.noop)
        if cond_prob == True:
            cond_prob = PostProcessor(Transform.get_lpsvm(3), Combiner.mul)
        assert isinstance(cond_prob, PostProcessor), type(cond_prob)

        assert X.shape[1] == self.nr_features, f"{X.shape[1]} != {self.nr_features}"
        if self.bias > 0:
            X = smat_util.append_column(X, self.bias)
        if not X.has_sorted_indices:
            X = X.sorted_indices()
        pX = PyMatrix.init_from(X, dtype=self.model_chain[0].pW.dtype)
        max_depth = min(self.depth, max_depth)
        pred_csr = csr_codes
        for d in range(max_depth):
            cur_model = self.model_chain[d]
            local_only_topk = only_topk if d == (max_depth - 1) else beam_size
            pred_csr = cur_model.predict(pX, only_topk=local_only_topk, csr_codes=pred_csr, cond_prob=cond_prob, threads=threads,)
        if normalized:
            pred_csr = sk_normalize(pred_csr, axis=1, copy=False, norm="l1")
        return pred_csr

    def predict_new(
        self, X, only_topk=None, csr_codes=None, beam_size=2, max_depth=None, cond_prob=True, normalized=False, threads=-1,
    ):
        if max_depth is None:
            max_depth = self.depth
        if cond_prob is None or cond_prob == False:
            cond_prob = PostProcessor(Transform.identity, Combiner.noop)
        if cond_prob == True:
            cond_prob = PostProcessor(Transform.get_lpsvm(3), Combiner.mul)
        assert isinstance(cond_prob, PostProcessor), tpye(cond_prob)

        assert X.shape[1] == self.nr_features
        if self.bias > 0:
            X = smat_util.append_column(X, self.bias)
        pX = PyMatrix.init_from(X, dtype=self.model_chain[0].pW.dtype)
        max_depth = min(self.depth, max_depth)
        pred_csr = csr_codes
        for d in range(max_depth):
            cur_model = self.model_chain[d]
            local_only_topk = only_topk if d == (max_depth - 1) else beam_size
            pred_csr = cur_model.predict_new(pX, only_topk=local_only_topk, csr_codes=pred_csr, cond_prob=cond_prob, threads=threads,)
        if normalized:
            pred_csr = sk_normalize(pred_csr, axis=1, copy=False, norm="l1")
        return pred_csr


class Parabel(object):
    """An utility Class to load model/prediction from Parabel Package"""

    @staticmethod
    def load_tree(path_to_file, path_to_param=None, bias=None):
        if path_to_param is not None:
            with open(path_to_param, "r") as fin:
                real_nr_features = int(fin.readline())
                for i in range(6):  # bypassing uncessary features
                    fin.readline()
                bias = float(fin.readline)
        if bias is None:
            bias = 1.0  # the bias term is default to 1.0 in the parabel package
        """Load a single tree model obtained from Parabel Package"""
        with open(path_to_file, "r") as fin:
            nr_features = int(fin.readline()) - 1 if bias <= 0 else 0  # remove the bias term
            nr_labels = int(fin.readline())
            nr_nodes = int(fin.readline())
            max_depth = int(sp.log2(nr_nodes + 1))
            Clist, Wlist = [], []
            for depth in range(max_depth):
                nr_nodes_with_depth = 2 ** depth
                if depth != max_depth - 1:
                    C = smat_util.coo_appender((2 ** (depth + 1), 2 ** depth))
                    W = smat_util.coo_appender((nr_features, 2 ** (depth + 1)))
                else:
                    C = smat_util.coo_appender((nr_labels, 2 ** depth))
                    W = smat_util.coo_appender((nr_features, nr_labels))

                child_offset = 2 ** (depth + 1) - 1
                for nid in range(nr_nodes_with_depth):
                    is_leaf = int(fin.readline().strip())
                    left, right = [int(x) - child_offset for x in fin.readline().strip().split()]
                    cur_depth = int(fin.readline().strip())
                    assert cur_depth == depth
                    tmp = fin.readline().strip().split()
                    labels = [int(y) for y in tmp[1:]]
                    nr_childs = int(fin.readline().strip().split()[0])
                    if is_leaf != 1:
                        labels = [left, right]
                    for y in labels:
                        C.append(y, nid, 1.0)
                        for iv in fin.readline().strip().split():
                            iv = iv.split(":")
                            col = y
                            row = int(iv[0])
                            if row >= nr_features:
                                continue
                            v = float(iv[1])
                            W.append(row, col, v)
                Clist += [C.tocsr()]
                Wlist += [W.tocsr()]
        return HierarchicalMLModel([MLModel(w, c) for w, c in zip(Wlist, Clist)], bias)

    @staticmethod
    def load_prediction(path_to_file, only_topk=None):
        with open(path_to_file, "r") as fin:
            nr_insts, nr_labels = [int(x) for x in fin.readline().strip().split()]
            coo = smat_util.coo_appender((nr_insts, nr_labels))
            for i in range(nr_insts):
                for iv in fin.readline().strip().split():
                    iv = iv.split(":")
                    j = int(iv[0])
                    v = float(iv[1])
                    coo.append(i, j, v)
        return smat_util.sorted_csr(coo.tocsr(), only_topk=only_topk)


class CountModel(object):
    def __init__(self, code_to_label):
        assert isinstance(code_to_label, smat.spmatrix)
        code_to_label = code_to_label.tocsr()
        self.code_to_label = sk_normalize(code_to_label, axis=1, copy=False, norm="l1")

    @property
    def nr_labels(self):
        return self.code_to_label.shape[1]

    @property
    def nr_codes(self):
        return self.code_to_label.shape[0]

    @classmethod
    def train(cls, prob, *arg_kw):
        assert prob.C is not None, "prob.C must be provided in CountModel.train()"
        return cls(prob.Z.T.dot(prob.Y))

    def predict(
        self, X, csr_codes=None, only_topk=None, cond_prob=True, normalize=False, **arg_kw,
    ):
        assert csr_codes is not None, "csr_codes must be provided for CountModel.prdict)"
        assert csr_codes.shape[0] == X.shape[0]
        assert csr_codes.shape[1] == self.nr_codes
        if cond_prob:
            pred_csr = csr_codes.dot(self.code_to_label).tocsr()
        else:
            tmp = csr_codes.data
            tmp2 = sp.ones_like(tmp)
            csr_codes.data = tmp2
            pred_csr = csr_codes.dot(self.code_to_label).tocsr()
            csr_codes.data = tmp

        pred_csr = smat_util.sorted_csr(pred_csr, only_topk=only_topk)
        if normalize:
            pred_csr = sk_normalize(pred_csr, axis=1, copy=False, norm="l1")
        return pred_csr


class CsrEnsembler(object):
    """A class implementing serveal ensembler for a list sorted CSR predictions"""

    @staticmethod
    def check_validlity(*args):
        for x in args:
            assert isinstance(x, smat.csr_matrix), type(x)
        assert all(x.shape == args[0].shape for x in args)

    @staticmethod
    def average(*args):
        CsrEnsembler.check_validlity(*args)
        ret = sum(args)
        ret = smat_util.sorted_csr(ret)
        ret.data /= len(args)
        return ret

    @staticmethod
    def rank_average(*args):
        CsrEnsembler.check_validlity(*args)
        mm = max((x.indptr[1:] - x.indptr[:-1]).max() for x in args)
        ret = sum(smat_util.get_relevance_csr(csr, mm) for csr in args)
        ret = smat_util.sorted_csr(ret)
        ret.data /= len(args)
        return ret

    @staticmethod
    def round_robin(*args):
        CsrEnsembler.check_validlity(*args)
        base = 1.0 / (len(args) + 1.0)
        mm = max((x.indptr[1:] - x.indptr[:-1]).max() for x in args)
        ret = smat_util.get_relevance_csr(args[0], mm)
        ret.data[:] += len(args) * base
        for i, x in enumerate(args[1:], 1):
            tmp = smat_util.get_relevance_csr(x, mm)
            tmp.data[:] += (len(args) - i) * base
            ret = ret.maximum(tmp)
        ret = smat_util.sorted_csr(ret)
        ret.data /= len(args)
        return ret

    @staticmethod
    def print_ens(Ytrue, pred_set, param_set):
        for param, pred in zip(param_set, pred_set):
            print("param: {}".format(param))
            print(Metrics.generate(Ytrue, pred))
        for ens in [
            CsrEnsembler.average,
            CsrEnsembler.rank_average,
            CsrEnsembler.round_robin,
        ]:
            print("ens: {}".format(ens.__name__))
            print(Metrics.generate(Ytrue, ens(*pred_set)))


def ml_train(X, Y, C=None, bias=None, hierarchical=None, min_labels=2, nr_splits=2, **arg_kw):
    """An interface function for HierarchicalMLModel.train"""
    prob = MLProblem(X, Y, C)
    return HierarchicalMLModel.train(prob, hierarchical, min_labels, nr_splits, bias=bias, **arg_kw)


def load_model(folder, dtype=None):
    if dtype is None:
        dtype = sp.float32
    param = json.loads(open("{}/param.json".format(folder), "r").read())
    cls = getattr(sys.modules[__name__], param["model"])
    return cls.load(folder, dtype=dtype)


def get_optimal_codes(Y, C, only_topk=None):
    csr_codes = smat_util.sorted_csr(Y.dot(C).tocsr(), only_topk=only_topk)
    csr_codes = sk_normalize(csr_codes, axis=1, copy=False, norm="l1")
    return csr_codes


# ============= Section for Ad-hoc Testing Code ==============
class Data(object):
    def __init__(self, X, Y, L, C, code, Xt=None, Yt=None, Xv=None, Yv=None, dataset=None):
        self.X = X  # feature matrix:  nr_insts * nr_features
        self.Y = Y  # label matrix:    nr_insts * nr_labels
        self.L = L  # label embedding: nr_labels * nr_label_features
        self.C = C  # label codes:     nr_labels * nr_codes
        self.code = code
        self.Xt = Xt
        self.Yt = Yt
        self.Xv = Xv
        self.Yv = Yv
        self.data_folder = "./datasets/{}".format(dataset)
        self.save_folder = "./save_models/{}".format(dataset)

    def update_codes(
        self, label_emb="elmo", kdim=2, depth=6, algo=indexer.Indexer.KMEANS, seed=0, max_iter=20, threads=-1, **arg_kw,
    ):

        # print('depth {} kdim {} label_emb {} algo {}'.format(depth, kdim, label_emb, algo))
        param = {
            "label_emb": label_emb,
            "depth": depth,
            "algo": algo,
            "seed": seed,
            "max_iter": max_iter,
        }
        code_name = "#".join(["{}:{}".format(k, v) for k, v in sorted(param.items())])
        code_npz = "{}/indexer/codes.{}.npz".format(self.save_folder, code_name)
        if path.exists(code_npz):
            self.C = smat.load_npz(code_npz)
        else:
            self.L = smat.load_npz("{}/L.{}.npz".format(self.data_folder, label_emb))
            code = indexer.Indexer(self.L).gen(kdim=kdim, depth=depth, algo=algo, seed=seed, max_iter=max_iter, threads=threads,)
            self.C = code.get_csc_matrix()
            smat.save_npz(code_npz, self.C, compressed=False)

    @classmethod
    def load(
        cls,
        dataset=None,
        label_emb="elmo",
        kdim=2,
        depth=6,
        algo=indexer.Indexer.KMEANS,
        seed=0,
        max_iter=10,
        threads=-1,
        dtype=None,
        **arg_kw,
    ):
        if dtype is None:
            dtype = sp.float32
        data_folder = "./datasets"
        X = smat.load_npz("{}/{}/X.trn.npz".format(data_folder, dataset))
        Y = smat.load_npz("{}/{}/Y.trn.npz".format(data_folder, dataset))
        try:
            Xt = smat.load_npz("{}/{}/X.tst.npz".format(data_folder, dataset))
            Yt = smat.load_npz("{}/{}/Y.tst.npz".format(data_folder, dataset))
            Xv = smat.load_npz("{}/{}/X.val.npz".format(data_folder, dataset))
            Yv = smat.load_npz("{}/{}/Y.val.npz".format(data_folder, dataset))
        except:
            Xt = None
            Yt = None
            Xv = None
            Yv = None
        L, code, C = None, None, None
        ret = cls(X, Y, L, C, code, Xt, Yt, Xv, Yv, dataset)
        if label_emb is not None:
            ret.update_codes(
                label_emb=label_emb, kdim=kdim, depth=depth, seed=seed, max_iter=max_iter, threads=threads,
            )
        return ret


def grid_search(data, grid_params, **kw_args):
    params = []
    results = []
    keys = list(grid_params.keys())
    for values in itertools.product(*[grid_params[k] for k in keys]):
        new_kw_args = kw_args.copy()
        new_kw_args.update(dict(zip(keys, values)))
        data.update_codes(**new_kw_args)
        prob = MLProblem(data.X, data.Y, data.C)
        model = ml_train(X=data.X, Y=data.Y, C=data.C, hierarchical=True, threshold=0.01, **new_kw_args,)
        pred_csr = model.predict(data.Xt, only_topk=20, beam_size=10, normalized=False)
        # print(Metrics.generate(data.Yt, pred_csr))
        results += [pred_csr]
        params += [dict(zip(keys, values))]
    return results, params


def test_speed(datafolder="dataset/Eurlex-4K", depth=3):
    data = Data.load(datafolder, depth=depth)
    X = data.X
    Y = data.Y
    C = data.C
    only_topk = 20
    topk = 10
    Cp = 1
    Cn = 1
    threshold = 0.01
    # solver_type = L2R_LR_DUAL
    solver_type = L2R_L2LOSS_SVC_DUAL
    # test multi-label with codes
    prob = MLProblem(X, Y, C)
    m = MLModel(smat.rand(data.X.shape[1], data.Y.shape[1], 0.1))
    rows = sp.arange(data.Yt.shape[0], dtype=sp.uint32)
    cols = sp.arange(data.Yt.shape[1], dtype=sp.uint32)
    inst_idx = sp.repeat(rows, sp.ones_like(rows, dtype=rows.dtype) * data.Yt.shape[1]).astype(sp.uint32)
    label_idx = sp.ones((len(rows), 1), dtype=sp.uint32).dot(cols.reshape(1, -1))[:]
    yy = m.predict_values(data.Xt, inst_idx, label_idx).reshape(data.Yt.shape[0], -1)


def test_svm(datafolder="dataset/Eurlex-4K", depth=3):
    data = Data.load(datafolder, depth=depth)
    X = PyMatrix(data.X, dtype=data.X.dtype)
    # X = data.X
    Y = data.Y
    C = data.C
    only_topk = 20
    topk = 10
    Cp = 1
    Cn = 1
    threshold = 0.01
    # solver_type = L2R_LR_DUAL
    solver_type = L2R_L2LOSS_SVC_DUAL

    # test multi-label with codes
    prob = MLProblem(X, Y, C)
    m = MLModel.train(prob, threshold=threshold, solver_type=solver_type, Cp=Cp, Cn=Cn)
    pred_Y = m.predict(X, only_topk=only_topk)
    print("sparse W with top {}".format(topk))
    metric = Metrics.generate(Y, pred_Y, topk)
    print(metric)
    """
    print('|W|^2 = {}'.format((m.W.toarray() * m.W.toarray()).sum()))
    coo = smat_util.dense_to_coo(sp.ones(pred_Y.shape))
    YY = smat_util.sorted_csr(smat.csr_matrix(m.predict_values(X, coo.row, coo.col).reshape(pred_Y.shape)))
    metric = Metrics.generate(Y, YY, topk)
    print(metric)
    YY = smat_util.sorted_csr(smat.csr_matrix(X.dot(m.W)))
    metric = Metrics.generate(Y, YY, topk)
    print(metric)
    """

    # test hierarchical multi-label
    print("Hierarchical-Multilabel")
    beam_size = 4
    min_labels = 2
    nr_splits = 2
    m = ml_train(prob, hierarchical=True, min_labels=min_labels, threshold=threshold, solver_type=solver_type, Cp=Cp, Cn=Cn,)
    print("m.depth = {}".format(m.depth))
    pred_Y = m.predict(X, beam_size=beam_size, only_topk=only_topk)
    print(pred_Y.shape)
    print("sparse W with top {}".format(topk))
    metric = Metrics.generate(Y, pred_Y, topk)
    print(metric)
    """
    max_depth = 2
    print('Predict up to depth = {}'.format(max_depth))
    pred_Y = m.predict(X, only_topk=only_topk, max_depth=max_depth)
    trueY = Y.copy()
    for d in range(m.depth - 1, max_depth - 1, -1):
        trueY = trueY.dot(m.model_chain[d].C)
    metric = Metrics.generate(trueY, pred_Y, topk)
    print(metric)
    #print('|W|^2 = {}'.format((m.W.toarray() * m.W.toarray()).sum()))
    """

    # test pure multi-label
    print("pure one-vs-rest Multi-label")
    prob = MLProblem(X, Y)
    m = MLModel.train(prob, threshold=threshold, solver_type=solver_type, Cp=Cp, Cn=Cn)
    pred_Y = m.predict(X, only_topk=only_topk)
    metric = Metrics.generate(Y, pred_Y, topk)
    print(metric)
    print("|W|^2 = {}".format((m.W.toarray() * m.W.toarray()).sum()))


if __name__ == "__main__":
    test_svm(datafolder="./datasets/Eurlex-4K", depth=6)
    test_speed(datafolder="./datasets/Eurlex-4K", depth=6)
