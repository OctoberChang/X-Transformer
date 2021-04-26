# -*- coding: utf-8 -*-

import sys
from os import path, system
from glob import glob
import scipy as sp
import scipy.sparse as smat
from scipy.sparse import identity as speye
import ctypes
from ctypes import *


def genFields(names, types):
    return list(zip(names, types))


def fillprototype(f, restype, argtypes):
    f.restype = restype
    f.argtypes = argtypes


def load_dynamic_library(dirname, soname, forced_rebuild=False):
    try:
        if forced_rebuild:
            system("make -C {} clean lib".format(dirname))
        path_to_so = glob(path.join(dirname, soname) + "*.so")[0]
        _c_lib = CDLL(path_to_so)
    except:
        try:
            system("make -C {} clean lib".format(dirname))
            path_to_so = glob(path.join(dirname, soname) + "*.so")[0]
            _c_lib = CDLL(path_to_so)
        except:
            raise Exception("{soname} library cannot be found and built.".format(soname=soname))
    return _c_lib


# Wrapper for Scipy/Numpy Matrix
class PyMatrix(ctypes.Structure):
    DENSE_ROWMAJOR = 1
    DENSE_COLMAJOR = 2
    SPARSE = 3
    EYE = 4

    _fields_ = [
        ("rows", c_uint64),
        ("cols", c_uint64),
        ("nnz", c_uint64),
        ("row_ptr", POINTER(c_uint64)),
        ("col_ptr", POINTER(c_uint64)),
        ("row_idx", POINTER(c_uint32)),
        ("col_idx", POINTER(c_uint32)),
        ("val", c_void_p),
        ("val_t", c_void_p),
        ("type", c_int32),
    ]

    def check_identiy(self, A):
        rows, cols = A.shape
        if rows != cols:
            return False
        if isinstance(A, sp.ndarray) and (sp.diag(A) == 1).all() != True:
            return False
        if isinstance(A, smat.spmatrix):
            return smat.csr_matrix(A) - speye(rows).nnz == 0

        return True

    @classmethod
    def identity(cls, size, dtype=sp.float32):
        eye = cls(A=None, dtype=dtype)
        eye.rows = c_uint64(size)
        eye.cols = c_uint64(size)
        eye.nnz = c_uint64(size)
        eye.dtype = dtype
        eye.type = PyMatrix.EYE
        name2type = dict(PyMatrix._fields_)
        for name in ["row_ptr", "col_ptr", "row_idx", "col_idx", "val", "val_t"]:
            setattr(eye, name, None)
        return eye

    def __init__(self, A, dtype=None):
        if A is None:
            return

        if dtype is None:
            dtype = sp.float32

        self.rows = c_uint64(A.shape[0])
        self.cols = c_uint64(A.shape[1])
        self.py_buf = {}
        self.dtype = dtype
        py_buf = self.py_buf

        if isinstance(A, (smat.csc_matrix, smat.csr_matrix)):
            Acsr = smat.csr_matrix(A)
            Acsc = smat.csc_matrix(A)
            self.type = PyMatrix.SPARSE
            self.nnz = c_uint64(Acsr.indptr[-1])
            py_buf["row_ptr"] = Acsr.indptr.astype(sp.uint64)
            py_buf["col_idx"] = Acsr.indices.astype(sp.uint32)
            py_buf["val_t"] = Acsr.data.astype(dtype)
            py_buf["col_ptr"] = Acsc.indptr.astype(sp.uint64)
            py_buf["row_idx"] = Acsc.indices.astype(sp.uint32)
            py_buf["val"] = Acsc.data.astype(dtype)

        elif isinstance(A, smat.coo_matrix):

            def coo_to_csr(coo):
                nr_rows, nr_cols, nnz, row, col, val = (
                    coo.shape[0],
                    coo.shape[1],
                    coo.data.shape[0],
                    coo.row,
                    coo.col,
                    coo.data,
                )
                indptr = sp.cumsum(sp.bincount(row + 1, minlength=(nr_rows + 1)), dtype=sp.uint64)
                indices = sp.zeros(nnz, dtype=sp.uint32)
                data = sp.zeros(nnz, dtype=dtype)
                sorted_idx = sp.argsort(row * sp.float64(nr_cols) + col)
                indices[:] = col[sorted_idx]
                data[:] = val[sorted_idx]
                return indptr, indices, data

            def coo_to_csc(coo):
                return coo_to_csr(smat.coo_matrix((coo.data, (coo.col, coo.row)), shape=[coo.shape[1], coo.shape[0]],))

            coo = A.tocoo()
            self.type = PyMatrix.SPARSE
            self.nnz = c_uint64(coo.data.shape[0])
            py_buf["row_ptr"], py_buf["col_idx"], py_buf["val_t"] = coo_to_csr(coo)
            py_buf["col_ptr"], py_buf["row_idx"], py_buf["val"] = coo_to_csc(coo)

        elif isinstance(A, sp.ndarray):
            py_buf["val"] = A.astype(dtype)
            if py_buf["val"].flags.f_contiguous:
                self.type = PyMatrix.DENSE_COLMAJOR
            else:
                self.type = PyMatrix.DENSE_ROWMAJOR
            self.nnz = c_uint64(A.shape[0] * A.shape[1])
        name2type = dict(PyMatrix._fields_)
        for name in py_buf:
            setattr(self, name, py_buf[name].ctypes.data_as(name2type[name]))
        self.buf = A

    @property
    def shape(self):
        return self.buf.shape

    def dot(self, other):
        return self.buf.dot(other)

    @classmethod
    def init_from(cls, A, dtype=None):
        if A is None:
            return None
        elif isinstance(A, PyMatrix):
            if dtype is None or A.dtype == dtype:
                return A
            else:
                return cls(A.buf, dtype)
        else:
            return cls(A, dtype)


class PredAllocator(object):
    CFUNCTYPE = CFUNCTYPE(None, c_uint64, c_uint64, c_uint64, c_void_p, c_void_p, c_void_p, c_void_p)

    def __init__(self, rows=0, cols=0, dtype=sp.float64):
        self.rows = rows
        self.cols = cols
        self.indptr = None
        self.indices = None
        self.data1 = None
        self.data2 = None
        self.dtype = dtype
        assert dtype == sp.float32 or dtype == sp.float64

    def __call__(self, rows, cols, nnz, indptr_ptr, indices_ptr, data1_ptr, data2_ptr):
        self.rows = rows
        self.cols = cols
        self.indptr = sp.zeros(self.cols + 1, dtype=sp.uint64)
        self.indices = sp.zeros(nnz, dtype=sp.uint64)
        self.data1 = sp.zeros(nnz, dtype=self.dtype)
        self.data2 = sp.zeros(nnz, dtype=self.dtype)

        cast(indptr_ptr, POINTER(c_uint64)).contents.value = self.indptr.ctypes.data_as(c_void_p).value
        cast(indices_ptr, POINTER(c_uint64)).contents.value = self.indices.ctypes.data_as(c_void_p).value
        cast(data1_ptr, POINTER(c_uint64)).contents.value = self.data1.ctypes.data_as(c_void_p).value
        cast(data2_ptr, POINTER(c_uint64)).contents.value = self.data2.ctypes.data_as(c_void_p).value

    def get_pred(self):
        csr_labels = smat.csc_matrix((self.data1, self.indices, self.indptr), shape=(self.rows, self.cols)).tocsr()
        pred_csr = smat.csc_matrix((self.data2, self.indices, self.indptr), shape=(self.rows, self.cols)).tocsr()
        return csr_labels, pred_csr

    @property
    def cfunc(self):
        return self.CFUNCTYPE(self)


class COOAllocator(object):
    CFUNCTYPE = CFUNCTYPE(None, c_uint64, c_uint64, c_uint64, c_void_p, c_void_p, c_void_p)

    def __init__(self, rows=0, cols=0, dtype=sp.float64):
        self.rows = rows
        self.cols = cols
        self.row_idx = None
        self.col_idx = None
        self.data = None
        self.dtype = dtype
        assert dtype == sp.float32 or dtype == sp.float64

    def __call__(self, rows, cols, nnz, row_ptr, col_ptr, val_ptr):
        self.rows = rows
        self.cols = cols
        self.row_idx = sp.zeros(nnz, dtype=sp.uint64)
        self.col_idx = sp.zeros(nnz, dtype=sp.uint64)
        self.data = sp.zeros(nnz, dtype=self.dtype)
        cast(row_ptr, POINTER(c_uint64)).contents.value = self.row_idx.ctypes.data_as(c_void_p).value
        cast(col_ptr, POINTER(c_uint64)).contents.value = self.col_idx.ctypes.data_as(c_void_p).value
        cast(val_ptr, POINTER(c_uint64)).contents.value = self.data.ctypes.data_as(c_void_p).value

    def tocoo(self):
        return smat.coo_matrix((self.data, (self.row_idx, self.col_idx)), shape=(self.rows, self.cols))

    def tocsr(self):
        return smat.csr_matrix((self.data, (self.row_idx, self.col_idx)), shape=(self.rows, self.cols))

    def tocsc(self):
        return smat.csc_matrix((self.data, (self.row_idx, self.col_idx)), shape=(self.rows, self.cols))

    @property
    def cfunc(self):
        return self.CFUNCTYPE(self)


class PyAllocator:
    CFUNCTYPE = CFUNCTYPE(c_long, c_int, POINTER(c_int), c_char)

    def __init__(self):
        self.allocated_arrays = []

    def __call__(self, dims, shape, dtype):
        x = sp.zeros(shape[:dims], sp.dtype(dtype))
        self.allocated_arrays.append(x)
        return x.ctypes.data_as(c_void_p).value

    def getcfunc(self):
        return self.CFUNCTYPE(self)

    cfunc = property(getcfunc)


class smat_util(object):
    class coo_appender(object):
        def __init__(self, shape):
            self.shape = shape
            self.row = []
            self.col = []
            self.val = []

        def append(self, i, j, v):
            self.row += [i]
            self.col += [j]
            self.val += [v]

        def tocoo(self):
            row = sp.array(self.row)
            col = sp.array(self.col)
            val = sp.array(self.val)
            return smat.coo_matrix((val, (row, col)), shape=self.shape)

        def tocsc(self):
            return self.tocoo().tocsc()

        def tocsr(self):
            return self.tocoo().tocsr()

    #"""
    @staticmethod
    def sorted_csr_from_coo(shape, row_idx, col_idx, val, only_topk=None):
        m = (sp.absolute(val.astype(sp.float64)).sum() + 1.0) * 3
        sorted_idx = sp.argsort(row_idx * m - val)
        row_idx[:] = row_idx[sorted_idx]
        col_idx[:] = col_idx[sorted_idx]
        val[:] = val[sorted_idx]
        indptr = sp.cumsum(sp.bincount(row_idx + 1, minlength=(shape[0] + 1)))
        if only_topk is not None and isinstance(only_topk, int):
            only_topk = max(min(1, only_topk), only_topk)
            selected_idx = (sp.arange(len(val)) - indptr[row_idx]) < only_topk
            row_idx = row_idx[selected_idx]
            col_idx = col_idx[selected_idx]
            val = val[selected_idx]
        indptr = sp.cumsum(sp.bincount(row_idx + 1, minlength=(shape[0] + 1)))
        return smat.csr_matrix((val, col_idx, indptr), shape=shape, dtype=val.dtype)
    #"""
    """
    @staticmethod
    def sorted_csr_from_coo(shape, row_idx, col_idx, val, only_topk=None):
        csr = smat.csr_matrix((val, (row_idx, col_idx)), shape=shape)
        csr.sort_indices()
        for i in range(shape[0]):
            rng = slice(csr.indptr[i], csr.indptr[i + 1])
            sorted_idx = sp.argsort(-csr.data[rng], kind="mergesort")
            csr.indices[rng] = csr.indices[rng][sorted_idx]
            csr.data[rng] = csr.data[rng][sorted_idx]
        if only_topk is not None:
            assert isinstance(only_topk, int), f"Wrong type: type(only_topk) = {type(only_topk)}"
            only_topk = max(min(1, only_topk), only_topk)
            nnz_of_insts = csr.indptr[1:] - csr.indptr[:-1]
            row_idx = sp.repeat(sp.arange(shape[0], dtype=sp.uint32), nnz_of_insts)
            selected_idx = (sp.arange(len(csr.data)) - csr.indptr[row_idx]) < only_topk
            row_idx = row_idx[selected_idx]
            col_idx = csr.indices[selected_idx]
            val = csr.data[selected_idx]
            indptr = sp.cumsum(sp.bincount(row_idx + 1, minlength=(shape[0] + 1)))
            csr = smat.csr_matrix((val, col_idx, indptr), shape=shape, dtype=val.dtype)
        return csr
    """
    @staticmethod
    def sorted_csc_from_coo(shape, row_idx, col_idx, val, only_topk=None):
        csr = smat_util.sorted_csr_from_coo(shape[::-1], col_idx, row_idx, val, only_topk=None)
        return smat.csc_matrix((csr.data, csr.indices, csr.indptr), shape, dtype=val.dtype)

    @staticmethod
    def sorted_csr(csr, only_topk=None):
        assert isinstance(csr, smat.csr_matrix)
        row_idx = sp.repeat(sp.arange(csr.shape[0], dtype=sp.uint32), csr.indptr[1:] - csr.indptr[:-1])
        return smat_util.sorted_csr_from_coo(csr.shape, row_idx, csr.indices, csr.data, only_topk)

    @staticmethod
    def sorted_csc(csc, only_topk=None):
        assert isinstance(csc, smat.csc_matrix)
        return smat_util.sorted_csr(csc.T).T

    @staticmethod
    def append_column(X, value=1.0, fast=True):
        assert len(X.shape) == 2
        new_column = value * sp.ones((X.shape[0], 1), dtype=X.dtype)
        if isinstance(X, smat.csc_matrix):
            if fast:  # around 5x to 10x faster than smat.hstack
                data = sp.concatenate((X.data, new_column.ravel()))
                indices = sp.concatenate((X.indices, sp.arange(X.shape[0], dtype=X.indices.dtype)))
                indptr = sp.concatenate((X.indptr, sp.array([X.indptr[-1] + X.shape[0]], dtype=X.indptr.dtype),))
                X = smat.csc_matrix((data, indices, indptr), shape=(X.shape[0], X.shape[1] + 1))
            else:
                X = smat.hstack([X, new_column]).tocsc()
        elif isinstance(X, smat.csr_matrix):
            if fast:  # around 5x to 10x faster than smat.hstack
                indptr = X.indptr + sp.arange(X.shape[0] + 1, dtype=X.indptr.dtype)
                indices = sp.zeros(len(X.indices) + X.shape[0], dtype=X.indices.dtype)
                data = sp.zeros(len(X.data) + X.shape[0], dtype=X.data.dtype)
                mask_loc = indptr[1:] - 1
                inv_mask = sp.ones_like(indices, dtype=sp.bool8)
                inv_mask[mask_loc] = False
                indices[mask_loc] = X.shape[1]
                data[mask_loc] = value
                indices[inv_mask] = X.indices
                data[inv_mask] = X.data
                X = smat.csr_matrix((data, indices, indptr), shape=(X.shape[0], X.shape[1] + 1))
            else:
                X = smat.hstack([X, new_column]).tocsr()
        elif isinstance(X, sp.ndarray):
            X = sp.hstack([X, new_column])
        return X

    @staticmethod
    def dense_to_coo(dense):
        rows = sp.arange(dense.shape[0], dtype=sp.uint32)
        cols = sp.arange(dense.shape[1], dtype=sp.uint32)
        row_idx = sp.repeat(rows, sp.ones_like(rows) * len(cols)).astype(sp.uint32)
        col_idx = sp.ones((len(rows), 1), dtype=sp.uint32).dot(cols.reshape(1, -1)).ravel()
        return smat.coo_matrix((dense.ravel(), (row_idx, col_idx)), shape=dense.shape)

    @staticmethod
    def get_relevance_csr(csr, mm=None, dtype=sp.float64):
        if mm is None:
            mm = (csr.indptr[1:] - csr.indptr[:-1]).max()
        nnz = len(csr.data)
        nnz_of_rows = csr.indptr[1:] - csr.indptr[:-1]
        row_idx = sp.repeat(sp.arange(csr.shape[0]), nnz_of_rows)
        rel = sp.array(mm - (sp.arange(nnz) - csr.indptr[row_idx]), dtype=dtype)  # adding 1 to avoiding zero entries
        return smat.csr_matrix((rel, csr.indices, csr.indptr), csr.shape)


def svm_read_problem(data_file_name, return_scipy=True):
    """
    svm_read_problem(data_file_name, return_scipy=False) -> [y, x], y: list, x: list of dictionary
    svm_read_problem(data_file_name, return_scipy=True)  -> [y, x], y: ndarray, x: csr_matrix

    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    scipy = sp
    prob_y = []
    prob_x = []
    row_ptr = [0]
    col_idx = []
    for i, line in enumerate(open(data_file_name)):
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1:
            line += [""]
        label, features = line
        prob_y += [float(label)]
        if scipy != None and return_scipy:
            nz = 0
            for e in features.split():
                ind, val = e.split(":")
                val = float(val)
                if val != 0:
                    col_idx += [int(ind) - 1]
                    prob_x += [val]
                    nz += 1
            row_ptr += [row_ptr[-1] + nz]
        else:
            xi = {}
            for e in features.split():
                ind, val = e.split(":")
                xi[int(ind)] = float(val)
            prob_x += [xi]
    if scipy != None and return_scipy:
        prob_y = scipy.array(prob_y)
        prob_x = scipy.array(prob_x)
        col_idx = scipy.array(col_idx)
        row_ptr = scipy.array(row_ptr)
        prob_x = smat.csr_matrix((prob_x, col_idx, row_ptr))
    return (prob_y, prob_x)
