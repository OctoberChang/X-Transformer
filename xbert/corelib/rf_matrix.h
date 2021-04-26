#ifndef RF_MATRIX_H
#define RF_MATRIX_H

// headers
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstddef>
#include <assert.h>
#include <omp.h>


#include <iostream>
#include <fstream>
#include <sstream>


#if __cplusplus >= 201103L || (defined(_MSC_VER) && (_MSC_VER >= 1500)) // Visual Studio 2008
#define CPP11
#endif

#ifdef _MSC_VER
#if _MSC_VER >= 1600
#include <cstdint>
#else
typedef __int8              int8_t;
typedef __int16             int16_t;
typedef __int32             int32_t;
typedef __int64             int64_t;
typedef unsigned __int8     uint8_t;
typedef unsigned __int16    uint16_t;
typedef unsigned __int32    uint32_t;
typedef unsigned __int64    uint64_t;
#endif
#else
#if !defined(_MSC_VER) && defined(CPP11)
#include <cstdint>
#else
typedef short int int16_t;
typedef int int32_t;
typedef long int64_t;
typedef unsigned char  uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;
#endif
#endif


/* random number genrator: simulate the interface of python random module*/
#include <limits>
#if defined(CPP11)
#include <random>
template<typename engine_t=std::mt19937>
struct random_number_generator : public engine_t {
    typedef typename engine_t::result_type result_type;

    random_number_generator(unsigned seed=0): engine_t(seed){ }

    result_type randrange(result_type end=engine_t::max()) { return engine_t::operator()() % end; }
    template<class T=double, class T2=double> T uniform(T start=0.0, T2 end=1.0) {
        return std::uniform_real_distribution<T>(start, (T)end)(*this);
    }
    template<class T=double> T normal(T mean=0.0, T stddev=1.0) {
        return std::normal_distribution<T>(mean, stddev)(*this);
    }
    template<class T=int, class T2=T> T randint(T start=0, T2 end=std::numeric_limits<T>::max()) {
        return std::uniform_int_distribution<T>(start, end)(*this);
    }
    template<class RandIter> void shuffle(RandIter first, RandIter last) {
        std::shuffle(first, last, *this);
    }
};
#else
#include <tr1/random>
template<typename engine_t=std::tr1::mt19937>
struct random_number_generator : public engine_t {
    typedef typename engine_t::result_type result_type;

    random_number_generator(unsigned seed=0): engine_t(seed) { }
    result_type operator()() { return engine_t::operator()(); }
    result_type operator()(result_type n) { return randint(result_type(0), result_type(n-1)); }

    result_type randrange(result_type end=engine_t::max()) { return engine_t::operator()() % end; }
    template<class T, class T2> T uniform(T start=0.0, T2 end=1.0) {
        typedef std::tr1::uniform_real<T> dist_t;
        return std::tr1::variate_generator<engine_t*, dist_t>(this, dist_t(start,(T)end))();
    }
    template<class T, class T2> T normal(T mean=0.0, T2 stddev=1.0) {
        typedef std::tr1::normal_distribution<T> dist_t;
        return std::tr1::variate_generator<engine_t*, dist_t>(this, dist_t(mean, (T)stddev))();
    }
    template<class T, class T2> T randint(T start=0, T2 end=std::numeric_limits<T>::max()) {
        typedef std::tr1::uniform_int<T> dist_t;
        return std::tr1::variate_generator<engine_t*, dist_t>(this, dist_t(start,end))();
    }
    template<class RandIter> void shuffle(RandIter first, RandIter last) {
        std::random_shuffle(first, last, *this);
    }
};
#endif
typedef random_number_generator<> rng_t;

template<typename T>
void gen_permutation_pair(size_t size, std::vector<T> &perm, std::vector<T> &inv_perm, int seed=0) {
    perm.resize(size);
    for(size_t i = 0; i < size; i++)
        perm[i] = i;

    rng_t rng(seed);
    rng.shuffle(perm.begin(), perm.end());
    //std::srand(seed);
    //std::random_shuffle(perm.begin(), perm.end());

    inv_perm.resize(size);
    for(size_t i = 0; i < size; i++)
        inv_perm[perm[i]] = i;
}

//#include "zlib_util.h"


#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))
#define CALLOC(type, size) (type*)calloc((size), sizeof(type))
#define REALLOC(ptr, type, size) (type*)realloc((ptr), sizeof(type)*(size))

typedef unsigned major_t;
const major_t ROWMAJOR = 1U;
const major_t COLMAJOR = 2U;
const major_t default_major = COLMAJOR;

// Zip Iterator
// Commom usage: std::sort(zip_iter(A.begin(),B.begin()), zip_iter(A.end(),B.end()));
template<class T1, class T2> struct zip_body;
template<class T1, class T2> struct zip_ref;
template<class IterT1, class IterT2> struct zip_it;
template<class IterT1, class IterT2> zip_it<IterT1, IterT2> zip_iter(IterT1 x, IterT2 y);

#define dvec_t dense_vector
template<typename val_type> class dvec_t;
#define svec_t sparse_vector
template<typename val_type> class svec_t;
#define sdvec_t sparse_dense_vector
template<typename val_type> class sdvec_t; // a dense vector with sparse indices
#define gvec_t general_vector
template<typename val_type> class gvec_t {
    public:
        size_t len;
        gvec_t(size_t len=0): len(len){}
        size_t size() const { return len; }
        virtual bool is_sparse() const {return false;}
        virtual bool is_dense() const {return false;}
        svec_t<val_type>& get_sparse() {assert(is_sparse()); return static_cast<svec_t<val_type>&>(*this);}
        const svec_t<val_type>& get_sparse() const {assert(is_sparse()); return static_cast<const svec_t<val_type>&>(*this);}
        dvec_t<val_type>& get_dense() {assert(is_dense()); return static_cast<dvec_t<val_type>&>(*this);}
        const dvec_t<val_type>& get_dense() const {assert(is_dense()); return static_cast<const dvec_t<val_type>&>(*this);}
};
#define dmat_t dense_matrix
template<typename val_type> class dmat_t;
#define smat_t sparse_matrix
template<typename val_type> class smat_t;
#define eye_t identity_matrix
template<typename val_type> class eye_t;
#define gmat_t general_matrix
template<typename val_type> class gmat_t {
    public:
        size_t rows, cols;
        gmat_t(size_t rows=0, size_t cols=0): rows(rows), cols(cols) { }

        size_t num_rows() const { return rows; }
        size_t num_cols() const { return cols; }
        virtual bool is_sparse() const { return false; }
        virtual bool is_dense() const { return false; }
        virtual bool is_identity() const { return false; }
        bool is_zero() const { return !is_sparse() && !is_dense() && !is_identity(); }

        smat_t<val_type>& get_sparse() { assert(is_sparse()); return static_cast<smat_t<val_type>&>(*this); }
        const smat_t<val_type>& get_sparse() const { assert(is_sparse()); return static_cast<const smat_t<val_type>&>(*this); }
        dmat_t<val_type>& get_dense() { assert(is_dense()); return static_cast<dmat_t<val_type>&>(*this); }
        const dmat_t<val_type>& get_dense() const { assert(is_dense()); return static_cast<const dmat_t<val_type>&>(*this); }

        virtual dvec_t<val_type>& Xv(const dvec_t<val_type>& v, dvec_t<val_type>& Xv, bool addson=0) const {
            if(addson == 0)
                memset(Xv.buf, 0, sizeof(val_type) * Xv.len);
            return Xv;
        }
        virtual dvec_t<val_type>& Xv(const svec_t<val_type>& v, dvec_t<val_type>& Xv, bool addson=0) const {
            if(addson == 0)
                memset(Xv.buf, 0, sizeof(val_type) * Xv.len);
            return Xv;
        }
        dvec_t<val_type>& Xv(const gvec_t<val_type>& v, dvec_t<val_type>& Xv, bool addson=0) const {
            if(v.is_sparse())
                return this->Xv(v.get_sparse(), Xv, addson);
            else if(v.is_dense())
                return this->Xv(v.get_dense(), Xv, addson);
            else // Should not be here
                return Xv;
        }

        virtual dvec_t<val_type>& XTu(const dvec_t<val_type>& u, dvec_t<val_type>& XTu, bool addson=0) const {
            if(addson == 0)
                memset(XTu.buf, 0, sizeof(val_type) * XTu.len);
            return XTu;
        }
        virtual dvec_t<val_type>& XTu(const svec_t<val_type>& u, dvec_t<val_type>& XTu, bool addson=0) const {
            if(addson == 0)
                memset(XTu.buf, 0, sizeof(val_type) * XTu.len);
            return XTu;
        }
        dvec_t<val_type>& XTu(const gvec_t<val_type>& u, dvec_t<val_type>& XTu, bool addson=0) const {
            if(u.is_sparse())
                return this->XTu(u.get_sparse(), XTu, addson);
            else if(u.is_dense())
                return this->XTu(u.get_dense(), XTu, addson);
            else // Should not be here
                return XTu;
        }
};
#define coo_t coo_matrix
template<typename val_type> class coo_t;

template<typename val_type> class entry_t;
template<typename val_type> class entry_iterator_t; // iterator base class
template<typename val_type> class file_iterator_t; // iterator for files with (i,j,v) tuples
template<typename val_type> class svmlight_file_iterator_t; // iterator for svmlight files
template<typename val_type> class coo_iterator_t; //iterator for three vectors (I, J, V) tuples
template<typename val_type> class smat_iterator_t; // iterator for nonzero entries in smat_t
template<typename val_type> class smat_subset_iterator_t; // iterator for nonzero entries in a subset
template<typename val_type> class dmat_iterator_t; // iterator for nonzero entries in dmat_t

/*------------------- Essential Linear Algebra Operations -------------------*/

// H = X*W, (X: m*n, W: n*k, H: m*k)
template<typename val_type> dmat_t<val_type>& dmat_x_dmat(const dmat_t<val_type> &X, const dmat_t<val_type> &W, dmat_t<val_type> &H);
template<typename val_type> dmat_t<val_type> operator*(const dmat_t<val_type> &X, const dmat_t<val_type> &W);
template<typename val_type> dmat_t<val_type>& smat_x_dmat(const smat_t<val_type>& X, const dmat_t<val_type> &W, dmat_t<val_type> &H);
template<typename val_type> dmat_t<val_type>& gmat_x_dmat(const gmat_t<val_type>& X, const dmat_t<val_type> &W, dmat_t<val_type> &H);
template<typename val_type> dmat_t<val_type> operator*(const smat_t<val_type> &X, const dmat_t<val_type> &W);

// H = a*X*W + H0, (X: m*n, W: n*k, H: m*k)
template<typename val_type, typename T2> dmat_t<val_type>& dmat_x_dmat(T2 a, const dmat_t<val_type> &X, const dmat_t<val_type> &W, const dmat_t<val_type> &H0, dmat_t<val_type> &H);
template<typename val_type, typename T2> dmat_t<val_type>& smat_x_dmat(T2 a, const smat_t<val_type> &X, const dmat_t<val_type> &W, const dmat_t<val_type> &H0, dmat_t<val_type> &H);
template<typename val_type, typename T2> dmat_t<val_type>& gmat_x_dmat(T2 a, const gmat_t<val_type> &X, const dmat_t<val_type> &W, const dmat_t<val_type> &H0, dmat_t<val_type> &H);

// H = a*X*W + b*H0, (X: m*n, W: n*k, H: m*k)
template<typename val_type, typename T2, typename T3> dmat_t<val_type>& dmat_x_dmat(T2 a, const dmat_t<val_type> &X, const dmat_t<val_type> &W, T3 b, const dmat_t<val_type> &H0, dmat_t<val_type> &H);
template<typename val_type, typename T2, typename T3> dmat_t<val_type>& smat_x_dmat(T2 a, const smat_t<val_type> &X, const dmat_t<val_type> &W, T3 b, const dmat_t<val_type> &H0, dmat_t<val_type> &H);
template<typename val_type, typename T2, typename T3> dmat_t<val_type>& gmat_x_dmat(T2 a, const gmat_t<val_type> &X, const dmat_t<val_type> &W, T3 b, const dmat_t<val_type> &H0, dmat_t<val_type> &H);

// trace(W'*X*H)
template<typename val_type> val_type trace_dmat_T_smat_dmat(const dmat_t<val_type>& W, const smat_t<val_type>& X, const dmat_t<val_type>& H);

// trace(W'*diag(D)*H)
template<typename val_type> val_type trace_dmat_T_diag_dmat(const dmat_t<val_type>& W, const dvec_t<val_type>& D, const dmat_t<val_type>& H);

/*-------------- Essential Linear Algebra Solvers -------------------*/

// Solve AX = B using Cholesky Factorization (A: Positive Definite)
template<typename val_type> dmat_t<val_type> ls_solve_chol(const dmat_t<val_type>& A, const dmat_t<val_type>& B, bool A_as_workspace);

// Solve Ax = b using Cholesky Factorization (A: Positive Definite)
template<typename val_type> dvec_t<val_type> ls_solve_chol(const dmat_t<val_type>& A, const dvec_t<val_type>& b, bool A_as_workspace);

// SVD: A = USV'
template<typename val_type> void svd(const dmat_t<val_type>& A, dmat_t<val_type>& U, dvec_t<val_type>& S, dmat_t<val_type>& V, bool reduced=true, bool A_as_workspace=false);

/*-------------- Vectors & Matrices -------------------*/

// Dense Vector
template<typename val_type>
class dvec_t : public gvec_t<val_type> {
    friend class dmat_t<val_type>;
    private:
        bool mem_alloc_by_me;
        void zero_init() {
            len = 0;
            buf = NULL;
            mem_alloc_by_me = false;
        }
    public:
        // size_t len; inherited from gvec_t
        using gvec_t<val_type>::len;
        val_type *buf;

        // Default Constructor
        dvec_t() { zero_init(); }

        // Copy Constructor
        dvec_t(const dvec_t& v) {
            zero_init();
            *this = v;
        }

        // Copy Assignment
        dvec_t& operator=(const dvec_t& other) {
            if(this == &other) { return *this; }
            if(other.is_view()) {  // view to view copy
                if(mem_alloc_by_me) clear_space();
                memcpy(static_cast<void*>(this), &other, sizeof(dvec_t));
            } else { // deep to deep copy
                resize(other.size());
                memcpy(buf, other.buf, sizeof(val_type)*len);
            }
            return *this;
        }

        // View Constructor: allocate space (w/ all 0) if buf == NULL
        explicit dvec_t(size_t len, val_type *buf=NULL): gvec_t<val_type>(len), mem_alloc_by_me(false), buf(buf) {
            if(buf == NULL && len != 0) {
                this->buf = MALLOC(val_type, len);
                memset(this->buf, 0, sizeof(val_type)*len);
                mem_alloc_by_me = true;
            }
        }

        // Fill Constructor
        explicit dvec_t(size_t len, const val_type &x) {
            zero_init();
            resize(len, x);
        }

        // Constructor - dense_matrix => dense_vector:
        //    Having the same status (view or deep) as m (the dense matrix).
        //    (expand the matrix using row major)
        dvec_t(const dmat_t<val_type>& m) {
            zero_init();
            if(m.is_view()) {
                len = m.rows * m.cols;
                buf = m.buf;
            }
            else {
                resize(m.rows * m.cols);
                memcpy(buf, m.buf, sizeof(val_type) * len);
            }
        }

        // Constructor - sparse_vector => dense_vector:
        //    Always deep.
        dvec_t(const svec_t<val_type>& v) {
            zero_init();
            resize(v.len);
            memset(buf, 0, sizeof(val_type) * len);
            for(size_t i = 0; i < v.nnz; i++)
                buf[v.idx[i]] = v.val[i];
        }

#if defined(CPP11)
        // Move Constructor
        dvec_t(dvec_t&& m) {
            zero_init();
            *this = std::move(m);
        }
        // Move Assignment
        dvec_t& operator=(dvec_t&& other) {
            if(this == &other) { return *this; }
            clear_space();
            memcpy(static_cast<void*>(this), &other, sizeof(dvec_t));
            other.zero_init();
            return *this;
        }
#endif
        ~dvec_t() { clear_space(); }

        bool is_view() const { return mem_alloc_by_me == false; }
        bool is_dense() const { return true; }

        void clear_space() {
            if(mem_alloc_by_me) { free(buf); }
            zero_init();
        }

        dvec_t get_view() const {
            return dvec_t(len, buf); // using view constructor
        }

        dvec_t& grow_body() {
            if(is_view()) {
                dvec_t tmp_view = *this; // Copy Assignment: View to view
                this->resize(len);
                memcpy(buf, tmp_view.buf, sizeof(val_type)*len);
            }
            return *this;
        }

        // Similar to operator=, but operator= uses view to view, deep to deep.
        // "assign" will directly change the underlying data, no matter view or deep.
        dvec_t& assign(const dvec_t& other) {
            assert(len == other.len);
            return assign((val_type)1.0, other);
        }

        // "assign" will directly change the underlying data, no matter view or deep.
        dvec_t& assign(val_type a, const dvec_t& other) {
            assert(len == other.len);
            if(a == val_type(0))
                memset(buf, 0, sizeof(val_type)*len);
            else if(a == val_type(1)) {
                if(this == &other)
                    return *this;
#pragma omp parallel for schedule(static)
                for(size_t idx = 0; idx < len; idx++)
                    at(idx) = other.at(idx);
            } else {
#pragma omp parallel for schedule(static)
                for(size_t idx = 0; idx < len; idx++)
                    at(idx) = a*other.at(idx);
            }
            return *this;
        }

        // resize will always grow body => is_view() becomes false
        void resize(size_t len_, const val_type &x) {
            resize(len_);
            if(x == 0)
                memset(buf, 0, sizeof(val_type) * len);
            else {
                std::fill_n(buf, len_, x);
                /*
                for(size_t i = 0; i < len; i++) {
                    buf[i] = x;
                }
                */
            }
        }

        // resize will always grow body => is_view() becomes false
        // (values in buf are not initialized)
        void resize(size_t len_) {
            if(mem_alloc_by_me)
                buf = REALLOC(buf, val_type, len_);
            else
                buf = MALLOC(val_type, len_);
            mem_alloc_by_me = true;
            len = len_;
        }

        val_type& at(size_t idx) { return buf[idx]; }
        const val_type& at(size_t idx) const { return buf[idx]; }

        val_type& operator[](size_t idx) { return buf[idx]; }
        const val_type& operator[](size_t idx) const { return buf[idx]; }

        val_type* data() { return buf; }
        const val_type* data() const { return buf; }

        val_type& back() { return buf[len - 1]; }
        const val_type& back() const { return buf[len - 1]; }

        void print(const char *str="") const {
            printf("%s dvec_t: len %lu, is_view %d, buf %p\n", str, len, is_view(), buf);
            for(size_t i = 0; i < len; i ++)
                printf("%.3f ", buf[i]);
            puts("");
        }

};

// Sparse Vector
template<typename val_type>
class svec_t : public gvec_t<val_type> {
    friend class smat_t<val_type>;
    private:
        bool mem_alloc_by_me;

        void zero_init() {
            len = nnz = 0;
            idx = NULL; val = NULL;
            mem_alloc_by_me = false;
        }
    public:
        // size_t len; inherited from gvec_t
        using gvec_t<val_type>::len;
        size_t nnz;
        unsigned *idx;
        val_type *val;

        // Default Constructor
        svec_t() { zero_init(); }

        // Copy Constructor
        svec_t(const svec_t& v) {
            zero_init();
            *this = v;
        }

        // Copy Assignment
        svec_t& operator=(const svec_t& other) {
            if(this == &other) return *this;
            if(other.is_view()) { // view to view copy
                if(mem_alloc_by_me) clear_space();
                memcpy(this, &other, sizeof(svec_t));
            } else { // deep to deep copy
                resize(other.len, other.nnz);
                memcpy(idx, other.idx, sizeof(unsigned) * nnz);
                memcpy(val, other.val, sizeof(val_type) * nnz);
            }
            return *this;
        }

        // View Constructor:
        //    If idx != NULL and val != NULL, we create a view copy. (view)
        //    Otherwise, we will allocate nnz space for both idx and val. (deep)
        explicit svec_t(size_t len, size_t nnz, unsigned *idx, val_type *val) : gvec_t<val_type>(len), mem_alloc_by_me(false), nnz(nnz) {
            if(nnz == 0){
                this->idx = NULL;
                this->val = NULL;
            }
            else {
                if(idx != NULL && val != NULL) {
                    this->idx = idx;
                    this->val = val;
                } else {
                    zero_init();
                    resize(len, nnz);
                }
            }
        }

        /* (Don't delete yet, so can understand codes not yet adapted elsewhere)
        // Fill Constructor:
        //    Always deep.
        //    If idx == NULL, we fill this->idx with 0.
        //    If idx != NULL, we still allocate this->idx and copy from idx.
        explicit svec_t(size_t len, size_t nnz, const unsigned *idx=NULL, const val_type &x=0) {
            zero_init();
            resize(len, nnz, x, idx);
        }
        */

        // Constructor - sparse_matrix => sparse_vector:
        //    Always deep. (expand using row major)
        svec_t(const smat_t<val_type>& m) {
            zero_init();
            resize(m.rows * m.cols, m.nnz);

            for(int i = 0; i < m.rows; i++) {
                for(int j = m.row_ptr[i]; j < m.row_ptr[i+1]; j++) {
                    idx[j] = m.cols * i + m.col_idx[j];
                    val[j] = m.val_t[j];
                }
            }
        }

        // Constructor - dense_vector => sparse_vector:
        //    Always deep.
        svec_t(const dvec_t<val_type>& v, double threshold=1e-12) {
            zero_init();
            len = v.size();
            for(size_t i = 0; i < v.size(); i++)
                if(fabs((double)v.at(i)) >= threshold)
                    nnz ++;
            resize(len, nnz);

            int k = 0;
            for(size_t i = 0; i < v.size(); i++)
                if(fabs((double)v.at(i)) >= threshold) {
                    idx[k] = i;
                    val[k] = v.at(i);
                    k++;
                }
        }

#if defined(CPP11)
        // Move Constructor
        svec_t(svec_t&& m) {
            zero_init();
            *this = std::move(m);
        }
        // Move Assignment
        svec_t& operator=(svec_t&& other) {
            if(this == &other) return *this;
            clear_space();
            memcpy(static_cast<void*>(this), &other, sizeof(svec_t));
            other.zero_init();
            return *this;
        }
#endif
        ~svec_t() { clear_space(); }

        size_t get_nnz() const { return nnz; }
        bool is_view() const { return mem_alloc_by_me == false; }
        bool is_sparse() const { return true; }

        void clear_space() {
            if(mem_alloc_by_me){
                free(idx);
                free(val);
            }
            zero_init();
        }

        svec_t get_view() const {
            return svec_t(len, nnz, idx, val); // using view constructor
        }

        svec_t& grow_body() {
            if(is_view()) {
                svec_t tmp_view = *this; // Copy Assignment: View to view
                this->resize(len, nnz);
                memcpy(idx, tmp_view.idx, sizeof(unsigned)*nnz);
                memcpy(val, tmp_view.val, sizeof(val_type)*nnz);
            }
            return *this;
        }

        // Similar to operator=, but operator= uses view to view, deep to deep.
        // "assign" will directly change the underlying data, no matter view or deep.
        // (so we assert that the sparse vector is not a view on sparse matrix)
        svec_t& assign(const svec_t& other) {
            assert(len == other.len && nnz == other.nnz);

            return assign((val_type)1.0, other);
        }

        // "assign" will directly change the underlying data, no matter view or deep.
        // (so we assert that the sparse vector is not a view on sparse matrix)
        svec_t& assign(val_type a, const svec_t& other) {
            assert(len == other.len && nnz == other.nnz);

            if(a == val_type(0))
                memset(val, 0, sizeof(val_type)*nnz);
            else if(a == val_type(1) && this == &other) {
                return *this;
            } else {
#pragma omp parallel for schedule(static)
                for(int k = 0; k < nnz; k++){
                    idx[k] = other.idx[k];
                    val[k] = a*other.val[k];
                }
            }
        }

        /* (Don't delete yet, so can understand codes not yet adapted elsewhere)
        // "resize" will always grow body => is_view() becomes false
        // (we will copy the whole idx to this->idx)
        void resize(size_t len_, size_t nnz_, const val_type &x, const unsigned *idx=NULL) {
            resize(len_, nnz_);
            if(idx == NULL)
                memset(this->idx, 0, sizeof(unsigned)*nnz);
            else
                memcpy(this->idx, idx, sizeof(unsigned)*nnz);

            for(size_t k = 0; k < nnz; k++)
                this->val[k] = x;
        }
        */

        // "resize" will always grow body => is_view() becomes false
        // (values in idx, val are not initialized)
        void resize(size_t len_, size_t nnz_) {
            if(mem_alloc_by_me){
                idx = REALLOC(idx, unsigned, nnz_);
                val = REALLOC(val, val_type, nnz_);
            }
            else{
                idx = MALLOC(unsigned, nnz_);
                val = MALLOC(val_type, nnz_);
            }
            mem_alloc_by_me = true;
            len = len_; nnz = nnz_;
        }

        void print(const char *str="") const {
            printf("%s svec_t: len %lu, nnz %lu, is_view %d\n", str, len, nnz, is_view());

            size_t j = 0;
            for(size_t i = 0; i < len; i++){
                if(j < nnz && idx[j] == i){
                    printf("%.3f ", val[j]);
                    j++;
                }
                else
                    printf("0.000 ");
            }
            puts("");
        }

};

// Sparse Dense Vector
template<typename val_type>
class sdvec_t : public dvec_t<val_type> {
    friend class smat_t<val_type>;
    public:
        using gvec_t<val_type>::len;
        using dvec_t<val_type>::buf;
        std::vector<unsigned> nz_idx;
        std::vector<unsigned char> is_nonzero;
        size_t nnz;
        sdvec_t(size_t len=0) :
            dvec_t<val_type>(len), nz_idx(len), is_nonzero(len), nnz(0){ }

        size_t get_nnz() const { return nnz; }

        void resize(size_t len_) {
            if(len != len_) {
                dvec_t<val_type>::resize(len_, 0.0);
                nz_idx.clear(); nz_idx.resize(len_);
                is_nonzero.clear(); is_nonzero.resize(len_);
                nnz = 0;
            }
        }

        template<typename V>
        void init_with_svec(const svec_t<V>& svec) {
            clear();
            nnz = svec.nnz;
            for(size_t t = 0; t < svec.nnz; t++) {
                size_t idx = svec.idx[t];
                V val = svec.val[t];
                is_nonzero[idx] = 1;
                nz_idx[t] = idx;
                buf[idx] = val;
            }
        }

        template<typename I, typename V>
        val_type& add_nonzero_at(I idx, V val) {
            buf[idx] += static_cast<val_type>(val);
            if(!is_nonzero[idx]) {
                is_nonzero[idx] = 1;
                nz_idx[nnz++] = static_cast<unsigned>(idx);
            }
            return buf[idx];
        }

        sdvec_t& update_nz_idx() {
            for(size_t t = 0 ; t < nnz; t++) {
                if(buf[nz_idx[t]] == static_cast<val_type>(0)) {
                    std::swap(nz_idx[t], nz_idx[nnz - 1]);
                    is_nonzero[nz_idx[t]] = 0;
                    t -= 1;
                    nnz -= 1;
                }
            }
            std::sort(nz_idx.data(), nz_idx.data() + nnz);
            nnz = std::unique(nz_idx.data(), nz_idx.data() + nnz) - nz_idx.data();
            return *this;
        }

        void clear() {
            if(nnz < (len >> 2)) {
                for(size_t t = 0; t < nnz; t++) {
                    buf[nz_idx[t]] = 0;
                    is_nonzero[nz_idx[t]] = 0;
                }
            } else {
                memset(buf, 0, sizeof(val_type) * len);
                memset(is_nonzero.data(), 0, sizeof(unsigned char) * len);
            }
            nnz = 0;
        }
};

// Dense Matrix
template<typename val_type>
class dmat_t : public gmat_t<val_type> {
    friend class dvec_t<val_type>;
    public:
        // size_t rows, cols; inherited from gmat_t
        using gmat_t<val_type>::rows;
        using gmat_t<val_type>::cols;
        val_type *buf;

        static dmat_t rand(rng_t &rng, size_t m, size_t n, double lower=0.0, double upper=1.0, major_t major_type_=default_major) {
            dmat_t ret(m, n, major_type_);
            if(lower >= upper) lower = upper;
            for(size_t idx = 0; idx < m*n; idx++)
                ret.buf[idx] = (val_type)rng.uniform(lower, upper);
            return ret;
        }

        static dmat_t randn(rng_t &rng, size_t m, size_t n, double mean=0.0, double std=1.0, major_t major_type_=default_major) {
            dmat_t ret(m, n, major_type_);
            for(size_t idx = 0; idx < m*n; idx++)
                ret.buf[idx] = (val_type)rng.normal(mean, std);
            return ret;
        }

    private:
        bool mem_alloc_by_me;
        major_t major_type;
        typedef dvec_t<val_type> vec_t;

        void zero_init() {
            rows = 0;
            cols = 0;
            buf = NULL;
            major_type = default_major;
            mem_alloc_by_me = false;
        }

    public:
        // Default Constructor
        dmat_t() { zero_init(); }

        // Copy Constructor:
        //    Having the same status (view or deep) as other.
        //    Using the same major_type as other.
        dmat_t(const dmat_t& other) {
            zero_init();
            *this = other;
        }

        // Copy Assignment:
        //    Having the same status (view or deep) as other.
        //    Using the same major_type as other.
        dmat_t& operator=(const dmat_t& other) {
            if(this == &other) return *this;
            if(other.is_view()) { // for view
                if(mem_alloc_by_me) clear_space();
                rows = other.rows;
                cols = other.cols;
                buf = other.buf;
                major_type = other.major_type;
                mem_alloc_by_me = false;
            } else { // deep copy
                if(is_view() || rows!=other.rows || cols!=other.cols || major_type!=other.major_type) {
                    major_type = other.major_type;
                    resize(other.rows, other.cols);
                }
                memcpy(buf, other.buf, sizeof(val_type)*rows*cols);
            }
            return *this;
        }

        // View Constructor:
        //    If buf != NULL, it creates a view on buf.
        //    If buf == NULL, it creates a deep matrix w/ all 0.
        explicit dmat_t(size_t rows_, size_t cols_, major_t major_type_=default_major, val_type *buf=NULL): gmat_t<val_type>(rows_,cols_), buf(buf), mem_alloc_by_me(false), major_type(major_type_) {
            if(buf == NULL && rows * cols != 0){
                this->buf = MALLOC(val_type, rows * cols);
                memset(this->buf, 0, sizeof(val_type) * rows * cols);
                mem_alloc_by_me = true;
            }
        }

        // Fill Constructor: fill in dense_vector based on the major_type.
        //    Always Deep.
        explicit dmat_t(size_t nr_copy, const dvec_t<val_type>& v, major_t major_type_=default_major) {
            zero_init();
            major_type = major_type_;
            resize(nr_copy, v);
        }

        // Constructor: dense_vector => dense_matrix:
        //    Having the same status (view or deep) as v (the dense vector).
        dmat_t(const dvec_t<val_type>& v, major_t major_type_=default_major) {
            zero_init();
            major_type = major_type_;
            if(!v.is_view())
                resize(1, v);
            else {
                rows = is_rowmajor()? 1: v.size();
                cols = is_colmajor()? 1: v.size();
                buf = v.buf;
            }
        }

        // Constructor: sparse_matrix => dense_matrix:
        //    Always deep.
        template<typename T>
        dmat_t(const smat_t<T>& sm, major_t major_type_=default_major) {
            zero_init();
            major_type = major_type_;
            resize(sm.rows, sm.cols);
            memset(buf, 0, sizeof(val_type)*rows*cols);
            for(size_t i = 0; i < sm.rows; i++)
                for(size_t idx = sm.row_ptr[i]; idx != sm.row_ptr[i+1]; idx++)
                    at(i, sm.col_idx[idx]) = sm.val_t[idx];
        }

        // Constructor: identity_matrix => dense_matrix:
        //    Always deep.
        template<typename T>
        dmat_t(const eye_t<T>& eye, major_t major_type_=default_major) {
            zero_init();
            major_type = major_type_;
            resize(eye.rows, eye.cols);
            memset(buf, 0, sizeof(val_type)*rows*cols);
            for(size_t i = 0; i < rows; i++)
                    at(i,i) = 1;
        }

#if defined(CPP11)
        // Move Constructor
        dmat_t(dmat_t&& m){
            zero_init();
            *this = std::move(m);
        }
        // Move Assignment
        dmat_t& operator=(dmat_t&& other) {
            if(this == &other) return *this;
            clear_space();
            rows = other.rows;
            cols = other.cols;
            buf = other.buf;
            mem_alloc_by_me = other.mem_alloc_by_me;
            major_type = other.major_type;
            other.zero_init();
            return *this;
        }
#endif
        ~dmat_t() { clear_space(); }

        bool is_view() const { return mem_alloc_by_me==false; }
        bool is_dense() const { return true; }
        bool is_rowmajor() const { return major_type==ROWMAJOR; }
        bool is_colmajor() const { return major_type==COLMAJOR; }

        major_t get_major() const { return major_type; }

        void clear_space() {
            if(mem_alloc_by_me) {
                free(buf);
            }
            zero_init();
        }

        // The view of the current dense matrix is returned.
        // (Using View Constructor)
        dmat_t get_view() const {
            return dmat_t(rows,cols,major_type,buf);
        }

        /* (Not yet deleted, to understand the behavior for unsync code elsewhere)
        // For ROWMAJOR, the view of a single row is returned.
        // For COLMAJOR, the view of a single column is returned.
        dvec_t<val_type> get_single_view(const size_t &idx) const {
            if(is_rowmajor())
                return dvec_t<val_type>(cols, &buf[idx * cols]);
            else
                return dvec_t<val_type>(rows, &buf[idx * rows]);
        }
        */

        // Return a view on the idx-th row of the dense matrix.
        // (Can only called when the matrix is ROWMAJOR)
        dvec_t<val_type> get_row(const size_t &idx) const {
            assert(is_rowmajor());

            if(is_rowmajor())
                return dvec_t<val_type>(cols, &buf[idx * cols]);
            else
                return dvec_t<val_type>();
        }

        // Return a view on the idx-th col of the dense matrix.
        // (Can only called when the matrix is COLMAJOR)
        dvec_t<val_type> get_col(const size_t &idx) const {
            assert(is_colmajor());

            if(is_colmajor())
                return dvec_t<val_type>(rows, &buf[idx * rows]);
            else
                return dvec_t<val_type>();
        }

        // For grow_body():
        //    Deep, View => Deep.
        // (this is the sole purpose of this function)
        dmat_t& grow_body() {
            if(is_view()) {
                dmat_t tmp_view = *this;
                this->resize(rows,cols);
                memcpy(buf, tmp_view.buf, sizeof(val_type) * rows * cols);
            }
            return *this;
        }

        // For transpose():
        //    It will return a view of the transpose of *this.
        //    (the major for ret will be the opposite of *this)
        dmat_t transpose() const {
            dmat_t ret = get_view();
            ret.to_transpose();
            return ret;
        }

        // ====================================================
        // ================ In-place functions ================
        // ====================================================

        // For assign():
        //    Deep => Deep.
        //    View => View.
        // Note: It differents from copy assignment!
        // After copy assignment, *this have the same status(View or Deep) as other.
        // But assign() directly overwrites the values in buf.
        // (it can modify the values it is viewing)
        dmat_t& assign(const dmat_t& other) {
            return assign((val_type)1.0, other);
        }

        // Similar to the above assign(), but now *this = a * other.
        template<typename T>
        dmat_t& assign(T a, const dmat_t& other) {
            if(a == T(0))
                memset(buf, 0, sizeof(val_type)*rows*cols);
            else if(a == T(1)) {
                if(this == &other)
                    return *this;
                if(is_rowmajor()) {
#pragma omp parallel for schedule(static)
                    for(size_t r = 0; r < rows; r++)
                        for(size_t c = 0; c < cols; c++)
                            at(r,c) = other.at(r,c);
                } else {
#pragma omp parallel for schedule(static)
                    for(size_t c = 0; c < cols; c++)
                        for(size_t r = 0; r < rows; r++)
                            at(r,c) = other.at(r,c);
                }
            } else {
                if(is_rowmajor()) {
#pragma omp parallel for schedule(static)
                    for(size_t r = 0; r < rows; r++)
                        for(size_t c = 0; c < cols; c++)
                            at(r,c) = a * other.at(r,c);
                } else {
#pragma omp parallel for schedule(static)
                    for(size_t c = 0; c < cols; c++)
                        for(size_t r = 0; r < rows; r++)
                            at(r,c) = a * other.at(r,c);
                }
            }
            return *this;
        }

        // After to_transpose():
        //    Deep => Deep.
        //    View => View.
        //    major_type will change.
        dmat_t& to_transpose() {
            std::swap(rows,cols);
            major_type = is_rowmajor()? COLMAJOR: ROWMAJOR;
            return *this;
        }

        // After inv_major():
        //    View, Deep => Deep.
        dmat_t& inv_major() {
            if(rows == cols && !is_view()) { // inplace for deep square matrix
                for(size_t r = 0; r < rows; r++)
                    for(size_t c = 0; c < r; c++)
                        std::swap(at(r,c),at(c,r));
                major_type = is_rowmajor()? COLMAJOR: ROWMAJOR;
            } else {
                dmat_t tmp(*this);
                major_type = is_rowmajor()? COLMAJOR: ROWMAJOR;
                resize(rows,cols);
                for(size_t r = 0; r < rows; r++)
                    for(size_t c = 0; c < cols; c++)
                        at(r,c) = tmp.at(r,c);
            }
            return *this;
        }

        // After to_rowmajor():
        //    Deep => Deep.
        //    View => View (if originally rowmajor), Deep (if originally colmajor).
        dmat_t& to_rowmajor() {
            if(is_colmajor()) inv_major();
            return *this;
        }

        // After to_colmajor():
        //    Deep => Deep.
        //    View => View (if originally colmajor), Deep (if originally rowmajor).
        dmat_t& to_colmajor() {
            if(is_rowmajor()) inv_major();
            return *this;
        }

        // After apply_permutation():
        //    Deep => Deep.
        //    View => View.
        // apply_permutation() directly overwrites the values in buf.
        // (thus it can modify the values dmat is viewing)
        dmat_t& apply_permutation(const std::vector<unsigned> &row_perm, const std::vector<unsigned> &col_perm) {
            return apply_permutation(row_perm.size()==rows? &row_perm[0]: NULL, col_perm.size()==cols? &col_perm[0] : NULL);
        }
        dmat_t& apply_permutation(const unsigned *row_perm=NULL, const unsigned *col_perm=NULL) {
            dmat_t tmp(*this);
            tmp.grow_body();
            for(size_t r = 0; r < rows; r++)
                for(size_t c = 0; c < cols; c++)
                    at(r,c) = tmp.at(row_perm? row_perm[r]: r, col_perm? col_perm[c]: c);
            return *this;
        }

        template<typename V1, typename V2>
        dmat_t& apply_scale(const V1 *row_scale, const V2 *col_scale) {
            if(row_scale != NULL && col_scale != NULL) {
                for(size_t r = 0; r < rows; r++) {
                    for(size_t c = 0; c < cols; c++) {
                        at(r, c) *= row_scale[r] * col_scale[c];
                    }
                }
            } else if(row_scale != NULL && col_scale == NULL) {
                for(size_t r = 0; r < rows; r++) {
                    for(size_t c = 0; c < cols; c++) {
                        at(r, c) *= row_scale[r];
                    }
                }
            } else if(row_scale == NULL && col_scale != NULL) {
                for(size_t r = 0; r < rows; r++) {
                    for(size_t c = 0; c < cols; c++) {
                        at(r, c) *= col_scale[c];
                    }
                }
            }
            return *this;
        }
        template<typename V>
        dmat_t& apply_scale(const dense_vector<V>& row_scale, const dense_vector<V>& col_scale) {
            return apply_scale(row_scale.data(), col_scale.data());
        }
        template<typename V>
        dmat_t& apply_row_scale(const dense_vector<V>& row_scale) {
            return apply_scale<V, V>(row_scale.data(), NULL);
        }
        template<typename V>
        dmat_t& apply_col_scale(const dense_vector<V>& col_scale) {
            return apply_scale<V, V>(NULL, col_scale.data());
        }

        // After resize():
        //    View, Deep => Deep.
        void resize(size_t nr_copy, const vec_t &v) {
            if(is_rowmajor()) {
                size_t rows_ = nr_copy, cols_ = v.size();
                resize(rows_, cols_);
                size_t unit = sizeof(val_type)*v.size();
                for(size_t r = 0; r < rows; r++)
                    memcpy(buf + r * cols, v.data(), unit);
            } else {
                size_t rows_ = v.size(), cols_ = nr_copy;
                resize(rows_, cols_);
                size_t unit = sizeof(val_type)*v.size();
                for(size_t c = 0; c < cols; c++)
                    memcpy(buf + c * rows, v.data(), unit);
            }
        }

        // After resize():
        //    View, Deep => Deep.
        dmat_t& resize(size_t rows_, size_t cols_) {
            if(mem_alloc_by_me) {
                if(rows_ == rows && cols_ == cols)
                    return *this;
                if(rows_*cols_ != rows*cols)
                    buf = REALLOC(buf, val_type, rows_*cols_);
            } else {
                buf = MALLOC(val_type, rows_*cols_);
            }

            mem_alloc_by_me = true;
            rows = rows_; cols = cols_;

            return *this;
        }

        // After lazy_resize():
        //   Deep => Deep.
        //   View => (If possible) ? View : Deep.
        dmat_t& lazy_resize(size_t rows_, size_t cols_, major_t major_type_=0) {
            if(is_view() && rows_*cols_==rows*cols &&
                    (major_type_ == 0 || major_type==major_type_))
                reshape(rows_,cols_);
            else {
                if(major_type_ != 0)
                    major_type = major_type_;
                resize(rows_, cols_);
            }
            return *this;
        }

        // After reshape:
        //    Deep => Deep.
        //    View => View.
        dmat_t& reshape(size_t rows_, size_t cols_) {
            assert(rows_*cols_ == rows*cols);
            if(rows_ != rows || cols != cols) {
                rows = rows_; cols = cols_;
            }
            return *this;
        }

        // ====================================================
        // ============ Dmat-Vector Multiplication ============
        // ====================================================

        dvec_t<val_type>& Xv(const dvec_t<val_type>& v, dvec_t<val_type>& Xv, bool addson=0) const {
            assert(v.size() == this->cols);
            if(Xv.size() != this->rows)
                Xv.resize(this->rows, 0.0);

            for(size_t i = 0; i < rows; i++) {
                if(addson == 0) Xv[i] = 0;
                for(size_t j = 0; j < cols; j++)
                    Xv[i] += at(i, j) * v[j];
            }
            return Xv;
        }
        dvec_t<val_type>& Xv(const svec_t<val_type>& v, dvec_t<val_type>& Xv, bool addson=0) const {
            assert(v.size() == this->cols);
            if(Xv.size() != this->rows)
                Xv.resize(this->rows, 0.0);

            for(size_t i = 0; i < rows; i++) {
                if(addson == 0) Xv[i] = 0;
                for(size_t p = 0; p < v.get_nnz(); p++)
                    Xv[i] += at(i, v.idx[p]) * v.val[p];
            }
            return Xv;
        }

        dvec_t<val_type>& XTu(const dvec_t<val_type>& u, dvec_t<val_type>& XTu, bool addson=0) const {
            assert(u.size() == this->rows);
            if(XTu.size() != this->cols)
                XTu.resize(this->rows, 0.0);

            for(size_t i = 0; i < cols; i++) {
                if(addson == 0) XTu[i] = 0;
                for(size_t j = 0; j < rows; j++)
                    XTu[i] += at(j, i) * u[j];
            }
            return XTu;
        }
        dvec_t<val_type>& XTu(const svec_t<val_type>& u, dvec_t<val_type>& XTu, bool addson=0) const {
            assert(u.size() == this->rows);
            if(XTu.size() != this->cols)
                XTu.resize(this->rows, 0.0);

            for(size_t i = 0; i < cols; i++) {
                if(addson == 0) XTu[i] = 0;
                for(size_t p = 0; p < u.get_nnz(); p++)
                    XTu[i] += at(u.idx[p], i) * u.val[p];
            }
            return XTu;
        }

        // ====================================================
        // ==================== IO Methods ====================
        // ====================================================

        void load_from_binary(const char *filename, major_t major_type_=default_major) {
            FILE *fp = fopen(filename, "rb");
            if(fp == NULL) {
                fprintf(stderr, "Error: can't read the file (%s)!!\n", filename);
                return;
            }
            load_from_binary(fp, major_type_, filename);
            fclose(fp);
        }
        void load_from_binary(FILE *fp, major_t major_type_=default_major, const char *filename=NULL) {
            clear_space();
            zero_init();

            size_t rows_, cols_;
            if(fread(&rows_, sizeof(size_t), 1, fp) != 1)
                fprintf(stderr, "Error: wrong input stream in %s.\n", filename);
            if(fread(&cols_, sizeof(size_t), 1, fp) != 1)
                fprintf(stderr, "Error: wrong input stream in %s.\n", filename);

            std::vector<double> tmp(rows_*cols_);
            if(fread(&tmp[0], sizeof(double), rows_*cols_, fp) != rows_*cols_)
                fprintf(stderr, "Error: wrong input stream in %s.\n", filename);

            dmat_t<double> tmp_view(rows_, cols_, ROWMAJOR, &tmp[0]);
            major_type = major_type_;
            resize(rows_, cols_);
            for(size_t r = 0; r < rows; r++)
                for(size_t c = 0; c < cols; c++)
                    at(r,c) = tmp_view.at(r,c);
        }
        void save_binary_to_file(const char *filename) {
            FILE *fp = fopen(filename, "wb");
            if(fp == NULL) {
                fprintf(stderr,"Error: can't open file %s\n", filename);
                exit(1);
            }
            save_binary_to_file(fp);
            fclose(fp);
        }
        void save_binary_to_file(FILE *fp) {
            fwrite(&rows, sizeof(size_t), 1, fp);
            fwrite(&cols, sizeof(size_t), 1, fp);
            std::vector<double> tmp(rows*cols);
            size_t idx = 0;
            for(size_t r = 0; r < rows; r++)
                for(size_t c = 0; c < cols; c++)
                    tmp[idx++] = (double)at(r,c);
            fwrite(&tmp[0], sizeof(double), tmp.size(), fp);
        }

        val_type& at(size_t r, size_t c) { return is_rowmajor()? buf[r*cols+c] : buf[c*rows+r]; }
        const val_type& at(size_t r, size_t c) const { return is_rowmajor()? buf[r*cols+c] : buf[c*rows+r]; }

        val_type* data() { return buf; }
        const val_type* data() const { return buf; }

        void print_mat(const char *str="", FILE *fp=stdout) const {
            fprintf(fp, "===>%s<===\n", str);
            fprintf(fp, "rows %ld cols %ld mem_alloc_by_me %d row_major %d\nbuf %p\n",
                    rows, cols, mem_alloc_by_me, is_rowmajor(), buf);
            for(size_t r = 0; r < rows; r++) {
                for(size_t c = 0; c < cols; c++)
                    fprintf(fp, "%.3f ", at(r,c));
                fprintf(fp, "\n");
            }
        }
};

// Identity Matrix
template<typename val_type>
class eye_t : public gmat_t<val_type> {
    public:
        // size_t rows, cols; inherited from gmat_t
        using gmat_t<val_type>::rows;
        using gmat_t<val_type>::cols;
        eye_t (size_t rows_ = 0) : gmat_t<val_type>(rows_, rows_){}
        bool is_identity() const { return true; }

        dvec_t<val_type>& Xv(const dvec_t<val_type>& v, dvec_t<val_type>& Xv, bool addson=0) const {
            assert(v.size() == this->cols);
            if(Xv.size() != this->rows)
                Xv.resize(this->rows, 0.0);

            return addson? do_axpy(1, v, Xv): Xv.assign(v);
        }
        dvec_t<val_type>& Xv(const svec_t<val_type>& v, dvec_t<val_type>& Xv, bool addson=0) const {
            assert(v.size() == this->cols);
            if(Xv.size() != this->rows)
                Xv.resize(this->rows, 0.0);

            dvec_t<val_type> dv(v);
            return addson? do_axpy(1, dv, Xv): Xv.assign(dv);
        }

        dvec_t<val_type>& XTu(const dvec_t<val_type>& u, dvec_t<val_type>& XTu, bool addson=0) const {
            assert(u.size() == this->rows);
            if(XTu.size() != this->cols)
                XTu.resize(this->rows, 0.0);

            return addson? do_axpy(1, u, XTu): XTu.assign(u);
        }
        dvec_t<val_type>& XTu(const svec_t<val_type>& u, dvec_t<val_type>& XTu, bool addson=0) const {
            assert(u.size() == this->rows);
            if(XTu.size() != this->cols)
                XTu.resize(this->rows, 0.0);

            dvec_t<val_type> du(u);
            return addson? do_axpy(1, du, XTu): XTu.assign(du);
        }
};

// Sparse Matrix (format CSC & CSR)
template<typename val_type>
class smat_t : public gmat_t<val_type> {
    private:
        bool mem_alloc_by_me;

        void zero_init() {
            mem_alloc_by_me = false;
            val=val_t=NULL;
            col_ptr=row_ptr=NULL;
            row_idx=col_idx=NULL;
            rows=cols=nnz=max_col_nnz=max_row_nnz=0;
        }

        void allocate_space(size_t rows_, size_t cols_, size_t nnz_) {
            if(mem_alloc_by_me)
                clear_space();

            rows = rows_; cols = cols_; nnz = nnz_;
            val = MALLOC(val_type, nnz);
            val_t = MALLOC(val_type, nnz);

            row_idx = MALLOC(unsigned, nnz);
            col_idx = MALLOC(unsigned, nnz);
            row_ptr = MALLOC(size_t, rows+1);
            col_ptr = MALLOC(size_t, cols+1);

            memset(row_ptr, 0, sizeof(size_t)*(rows+1));
            memset(col_ptr, 0, sizeof(size_t)*(cols+1));
            mem_alloc_by_me = true;
        }

        void csc_to_csr_old() {
            memset(row_ptr, 0, sizeof(size_t)*(rows+1));
            for(size_t idx = 0; idx < nnz; idx++)
                row_ptr[row_idx[idx]+1]++;
            for(size_t r = 1; r <= rows; r++)
                row_ptr[r] += row_ptr[r-1];
            for(size_t c = 0; c < cols; c++) {
                for(size_t idx = col_ptr[c]; idx != col_ptr[c+1]; idx++) {
                    size_t r = (size_t) row_idx[idx];
                    col_idx[row_ptr[r]] = c;
                    val_t[row_ptr[r]++] = val[idx];
                }
            }
            for(size_t r = rows; r > 0; r--)
                row_ptr[r] = row_ptr[r-1];
            row_ptr[0] = 0;
        }
        void csc_to_csr() {
            smat_t tmp = this->transpose();
            tmp.csr_to_csc();
        }
        void csr_to_csc() {
            memset(col_ptr, 0, sizeof(size_t) * (cols + 1));
            for(size_t idx = 0; idx < nnz; idx++) {
                col_ptr[col_idx[idx] + 1]++;
            }
            for(size_t c = 1; c <= cols; c++) {
                col_ptr[c] += col_ptr[c - 1];
            }
            for(size_t r = 0; r < rows; r++) {
                for(size_t idx = row_ptr[r]; idx != row_ptr[r + 1]; idx++) {
                    size_t c = (size_t) col_idx[idx];
                    row_idx[col_ptr[c]] = r;
                    val[col_ptr[c]++] = val_t[idx];
                }
            }
            for(size_t c = cols; c > 0; c--) {
                col_ptr[c] = col_ptr[c - 1];
            }
            col_ptr[0] = 0;
        }

        void update_max_nnz() {
            max_row_nnz = max_col_nnz = 0;
            for(size_t c = 0; c < cols; c++) max_col_nnz = std::max(max_col_nnz, nnz_of_col(c));
            for(size_t r = 0; r < rows; r++) max_row_nnz = std::max(max_row_nnz, nnz_of_row(r));
        }

        // Comparator for sorting rates into row/column comopression storage
        class SparseLess {
            public:
                const unsigned *row_idx;
                const unsigned *col_idx;
                SparseLess(const unsigned *row_idx_, const unsigned *col_idx_, bool isCSR=true) {
                    row_idx = (isCSR)? row_idx_: col_idx_;
                    col_idx = (isCSR)? col_idx_: row_idx_;
                }
                bool operator()(size_t x, size_t y) const {
                    return  (row_idx[x] < row_idx[y]) || ((row_idx[x] == row_idx[y]) && (col_idx[x] < col_idx[y]));
                }
        };
        class SparseEq {
            public:
                const unsigned *row_idx;
                const unsigned *col_idx;
                SparseEq(const unsigned *row_idx_, const unsigned *col_idx_) {
                    row_idx = row_idx_;
                    col_idx = col_idx_;
                }
                bool operator()(size_t x, size_t y) const {
                    return (row_idx[x] == row_idx[y]) && (col_idx[x] == col_idx[y]);
                }
        };

    public: // static methods
        static smat_t rand(rng_t &rng, size_t m, size_t n, double sparsity=0.01, double lower=0.0, double upper=1.0) {
            if(lower > upper) lower = upper;
            smat_t ret;
            size_t nnz_ = (size_t)(m*n*sparsity);
            ret.allocate_space(m, n, nnz_);
            for(size_t idx = 0; idx < nnz_; idx++) {
                ret.val_t[idx] = rng.uniform(lower, upper);
                ret.col_idx[idx] = rng.randint(0, n-1);
                ret.row_ptr[rng.randint(1, m)] += 1;
            }
            for(size_t i = 1; i <= m; i++)
                ret.row_ptr[i] += ret.row_ptr[i-1];
            ret.csr_to_csc();
            ret.update_max_nnz();
            return ret;
        }
        static smat_t randn(rng_t &rng, size_t m, size_t n, double sparsity=0.01, double mean=0.0, double std=1.0) {
            smat_t ret;
            size_t nnz_ = (size_t)(m*n*sparsity);
            ret.allocate_space(m, n, nnz_);
            for(size_t idx = 0; idx < nnz_; idx++) {
                ret.val_t[idx] = (val_type)rng.normal(mean, std);
                ret.col_idx[idx] = rng.randint(0, n-1);
                ret.row_ptr[rng.randint(1,m)] += 1;
            }
            for(size_t i = 1; i <= m; i++)
                ret.row_ptr[i] += ret.row_ptr[i-1];
            ret.csr_to_csc();
            ret.update_max_nnz();
            return ret;
        }

        // rows, cols are inherited from gmat_t
        using gmat_t<val_type>::rows;
        using gmat_t<val_type>::cols;
        size_t nnz, max_row_nnz, max_col_nnz;

        val_type *val, *val_t;
        size_t *col_ptr, *row_ptr;
        unsigned *row_idx, *col_idx;

        // filetypes for loading smat_t
        enum format_t { TXT=0, PETSc=1, SVMLIGHT=2, BINARY=3, COMPRESSION=4 };

        // Default Constructor
        smat_t() { zero_init(); }

        // Copy Constructor
        smat_t(const smat_t& m) {
            zero_init();
            *this = m;
        }

        // Copy Assignment
        // view => view, deep => deep.
        smat_t& operator=(const smat_t& other) {
            if(this == &other) { return *this; }
            if(mem_alloc_by_me) { clear_space(); }
            if(other.is_view()) { // for view
                memcpy(static_cast<void*>(this), &other, sizeof(smat_t));
            } else { // deep copy
                *this = other.get_view();
                grow_body();
            }
            return *this;
        }

        // View Constructor:
        explicit smat_t(size_t rows, size_t cols, size_t nnz,
                val_type *val, val_type *val_t,
                size_t *col_ptr, size_t *row_ptr,
                unsigned *row_idx, unsigned *col_idx) :
                gmat_t<val_type>(rows, cols), nnz(nnz),
                val(val), val_t(val_t),
                col_ptr(col_ptr), row_ptr(row_ptr),
                row_idx(row_idx), col_idx(col_idx)
                { mem_alloc_by_me = false; update_max_nnz(); }

        // Constructor: dense matrix => sparse matrix
        smat_t(const dmat_t<val_type>& m) {
            zero_init();
            dmat_iterator_t<val_type> entry_it(m);
            load_from_iterator(m.rows, m.cols, entry_it.get_nnz(), &entry_it);
        }

        // Constructor: identity matrix => sparse matrix
        smat_t(const eye_t<val_type>& eye) {
            zero_init();
            allocate_space(eye.rows, eye.rows, eye.rows);
            for(size_t i = 0; i < eye.rows; i++) {
                row_ptr[i+1] = i+1;
                col_idx[i] = i;
                val_t[i] = (val_type)1;
            }
            for(size_t j = 0; j < eye.cols; j++) {
                col_ptr[j+1] = j+1;
                row_idx[j] = j;
                val[j] = (val_type)1;
            }
        }

        smat_t(size_t rows_, size_t cols_, size_t nnz_=0){
            zero_init();
            allocate_space(rows_, cols_, nnz_);
        }

#if defined(CPP11)
        // Move Constructor
        smat_t(smat_t&& m){
            zero_init();
            *this = std::move(m);
        }
        // Move Assignment
        smat_t& operator=(smat_t&& other) {
            if(this == &other) { return *this; }
            clear_space();
            memcpy(static_cast<void*>(this), &other, sizeof(smat_t));
            other.zero_init();
            return *this;
        }
#endif
        // Destructor
        ~smat_t(){ clear_space(); }

        size_t get_nnz() const { return nnz; }
        bool is_view() const { return mem_alloc_by_me==false; }
        bool is_sparse() const { return true; }

        void clear_space() {
            if(mem_alloc_by_me) {
                if(val) { free(val); } if(val_t) { free(val_t); }
                if(row_ptr) { free(row_ptr); } if(row_idx) { free(row_idx); }
                if(col_ptr) { free(col_ptr); } if(col_idx) { free(col_idx); }
            }
            zero_init();
        }

        smat_t get_view() const {
            if(is_view()) {
                return *this;
            } else {
                smat_t tmp;
                memcpy(static_cast<void*>(&tmp), this, sizeof(smat_t));
                tmp.mem_alloc_by_me = false;
                return tmp;
            }
        }

        /* (Don't delete yet, so can understand codes not yet adapted elsewhere)
        svec_t<val_type> get_single_view(const size_t &idx, const major_t &major=default_major) const {
            if(major == ROWMAJOR)
                return svec_t<val_type>(cols, nnz_of_row(idx), &col_idx[row_ptr[idx]], &val_t[row_ptr[idx]], 0);
            else
                return svec_t<val_type>(rows, nnz_of_col(idx), &row_idx[col_ptr[idx]], &val[col_ptr[idx]], 0);
        }
        */

        // For get_row and get_col, a sparse vector view is returned.
        // Caveat: If you directly modify the returned sparse vector view,
        //         it will change the sparse matrix's underlying data.
        //         And because we store both column and row major format,
        //         the modification on the returned svec_t will only effect one of the format.
        //         Resulting in an inconsistency within the sparse matrix.
        // Summary: Do not directly modify the returned sparse vector view.
        //          (if the view becomes a deep vector afterwards, then things will be fine.)
        svec_t<val_type> get_row(const size_t &idx) const {
            return svec_t<val_type>(cols, nnz_of_row(idx), &col_idx[row_ptr[idx]], &val_t[row_ptr[idx]]);
        }
        svec_t<val_type> get_col(const size_t &idx) const {
            return svec_t<val_type>(rows, nnz_of_col(idx), &row_idx[col_ptr[idx]], &val[col_ptr[idx]]);
        }

        smat_t& grow_body() {
            if(is_view()) {
                smat_t tmp = *this; // a copy of the view
                col_ptr = MALLOC(size_t, cols + 1); memcpy(col_ptr, tmp.col_ptr, sizeof(size_t) * (cols + 1));
                row_idx = MALLOC(unsigned, nnz); memcpy(row_idx, tmp.row_idx, sizeof(unsigned) * nnz);
                val = MALLOC(val_type, nnz); memcpy(val, tmp.val, sizeof(val_type) * nnz);
                row_ptr = MALLOC(size_t, rows + 1); memcpy(row_ptr, tmp.row_ptr, sizeof(size_t) * (rows + 1));
                col_idx = MALLOC(unsigned, nnz); memcpy(col_idx, tmp.col_idx, sizeof(unsigned) * nnz);
                val_t = MALLOC(val_type, nnz); memcpy(val_t, tmp.val_t, sizeof(val_type) * nnz);
                mem_alloc_by_me = true;
            }
            return *this;
        }

        smat_t transpose() const{
            smat_t<val_type> mt = get_view().to_transpose();
            return mt;
        }

        // ====================================================
        // ================ In-place functions ================
        // ====================================================

        smat_t& to_transpose() {
            std::swap(rows,cols);
            std::swap(val,val_t);
            std::swap(row_ptr,col_ptr);
            std::swap(row_idx,col_idx);
            std::swap(max_col_nnz, max_row_nnz);
            return *this;
        }

        smat_t& apply_permutation(const std::vector<unsigned> &row_perm, const std::vector<unsigned> &col_perm) {
            return apply_permutation(row_perm.size()==rows? &row_perm[0]: NULL, col_perm.size()==cols? &col_perm[0]: NULL);
        }
        smat_t& apply_permutation(const unsigned *row_perm=NULL, const unsigned *col_perm=NULL) {
            if(row_perm != NULL) {
                for(size_t idx = 0; idx < nnz; idx++) {
                    row_idx[idx] = row_perm[row_idx[idx]];
                }
                csc_to_csr();
                csr_to_csc();
            }
            if(col_perm != NULL) {
                for(size_t idx = 0; idx < nnz; idx++) {
                    col_idx[idx] = col_perm[col_idx[idx]];
                }
                csr_to_csc();
                csc_to_csr();
            }
            return *this;
        }

        template<typename V1, typename V2>
        smat_t& apply_scale(const V1 *row_scale, const V2 *col_scale) {
            if(row_scale != NULL && col_scale != NULL) {
                for(size_t r = 0; r < rows; r++) {
                    val_type alpha = row_scale[r];
                    for(size_t idx = row_ptr[r]; idx != row_ptr[r + 1]; idx++) {
                        val_t[idx] *= alpha * col_scale[col_idx[idx]];
                    }
                }
                for(size_t c = 0; c < cols; c++) {
                    val_type alpha = col_scale[c];
                    for(size_t idx = col_ptr[c]; idx != col_ptr[c + 1]; idx++) {
                        val[idx] *= alpha * row_scale[row_idx[idx]];
                    }
                }
            } else if(row_scale != NULL && col_scale == NULL) {
                for(size_t r = 0; r < rows; r++) {
                    if(nnz_of_row(r)) {
                        for(size_t idx = row_ptr[r]; idx < row_ptr[r + 1]; idx++) {
                            val_t[idx] *= row_scale[r];
                        }
                    }
                }
                for(size_t idx = 0; idx < nnz; idx++) {
                    val[idx] *= row_scale[row_idx[idx]];
                }
            } else if(row_scale == NULL && col_scale != NULL) {
                for(size_t c = 0; c < cols; c++) {
                    if(nnz_of_col(c)) {
                        for(size_t idx = col_ptr[c]; idx < col_ptr[c + 1]; idx++) {
                            val[idx] *= col_scale[c];
                        }
                    }
                }
                for(size_t idx = 0; idx < nnz; idx++) {
                    val_t[idx] *= col_scale[col_idx[idx]];
                }
            }
            return *this;
        }
        template<typename V1, typename V2>
        smat_t& apply_scale(const dvec_t<V1> &row_scale, const dvec_t<V2> &col_scale) {
            return apply_scale(row_scale.data(), col_scale.data());
        }
        template<typename V>
        smat_t& apply_row_scale(const dvec_t<V> &row_scale) {
            return apply_scale<V, V>(row_scale.data(), NULL);
        }
        template<typename V>
        smat_t& apply_col_scale(const dvec_t<V> &col_scale) {
            return apply_scale<V, V>(NULL, col_scale.data());
        }

        smat_t row_subset(const std::vector<unsigned> &subset) const {
            return row_subset(&subset[0], (int)subset.size());
        }
        smat_t row_subset(const unsigned *subset, int subset_size) const {
            smat_subset_iterator_t<val_type> it(*this, subset, subset_size, ROWMAJOR);
            smat_t<val_type> sub_smat;
            sub_smat.load_from_iterator(subset_size, cols, it.get_nnz(), &it);
            return sub_smat;
        }
        smat_t col_subset(const std::vector<unsigned> &subset) const {
            return col_subset(&subset[0], (int)subset.size());
        }
        smat_t col_subset(const unsigned *subset, int subset_size) const {
            smat_subset_iterator_t<val_type> it(*this, subset, subset_size, COLMAJOR);
            smat_t<val_type> sub_smat;
            sub_smat.load_from_iterator(rows, subset_size, it.get_nnz(), &it);
            return sub_smat;
        }

        size_t nnz_of_row(unsigned i) const { return (row_ptr[i+1] - row_ptr[i]); }
        size_t nnz_of_col(unsigned i) const { return (col_ptr[i+1] - col_ptr[i]); }

        // ====================================================
        // ============ Smat-Vector Multiplication ============
        // ====================================================

        val_type* Xv(const val_type* v, val_type* Xv, bool addson=0) const {
            for(size_t i = 0; i < rows; i++) {
                if(addson == 0) Xv[i] = 0;
                for(size_t idx = row_ptr[i]; idx < row_ptr[i+1]; idx++)
                    Xv[i] += val_t[idx] * v[col_idx[idx]];
            }
            return Xv;
        }
        dvec_t<val_type>& Xv(const dvec_t<val_type>& v, dvec_t<val_type>& Xv, bool addson=0) const {
            assert(v.size() == this->cols);
            if(Xv.size() != this->rows)
                Xv.resize(this->rows, 0.0);

            this->Xv(v.data(), Xv.data(), addson);
            return Xv;
        }
        dvec_t<val_type>& Xv(const svec_t<val_type>& v, dvec_t<val_type>& Xv, bool addson=0) const {
            assert(v.size() == this->cols);
            if(Xv.size() != this->rows)
                Xv.resize(this->rows, 0.0);

            if(addson == 0) {
                for(size_t i = 0; i < Xv.size(); i++) {
                    Xv[i] = 0;
                }
            }

            for(size_t k = 0; k < v.nnz; k++) {
                size_t col_idx = static_cast<size_t>(v.idx[k]);
                const val_type& alpha = v.val[k];
                do_axpy(alpha, get_col(col_idx), Xv);
            }
            /* slower implementatoin
            dvec_t<val_type> dv(v);
            this->Xv(dv.data(), Xv.data(), addson);
            */
            return Xv;
        }

        val_type* XTu(const val_type* u, val_type* XTu, bool addson=0) const {
            for(size_t i = 0; i < cols; i++) {
                if(addson == 0) XTu[i] = 0;
                for(size_t idx = col_ptr[i]; idx < col_ptr[i+1]; idx++)
                    XTu[i] += val[idx] * u[row_idx[idx]];
            }
            return XTu;
        }
        dvec_t<val_type>& XTu(const dvec_t<val_type>& u, dvec_t<val_type>& XTu, bool addson=0) const {
            assert(u.size() == this->rows);
            if(XTu.size() != this->cols)
                XTu.resize(this->rows, 0.0);

            this->XTu(u.data(), XTu.data(), addson);
            return XTu;
        }
        dvec_t<val_type>& XTu(const svec_t<val_type>& u, dvec_t<val_type>& XTu, bool addson=0) const {
            assert(u.size() == this->rows);
            if(XTu.size() != this->cols)
                XTu.resize(this->rows, 0.0);

            if(addson == 0) {
                for(size_t i = 0; i < XTu.size(); i++) {
                    XTu[i] = 0;
                }
            }

            for(size_t k = 0; k < u.nnz; k++) {
                size_t row_idx = static_cast<size_t>(u.idx[k]);
                const val_type& alpha = u.val[k];
                do_axpy(alpha, get_row(row_idx), XTu);
            }

            /* slower implementatoin
            dvec_t<val_type> du(u);
            this->XTu(du.data(), XTu.data(), addson);
            */
            return XTu;
        }

        // ====================================================
        // ==================== IO Methods ====================
        // ====================================================

        // The entry_iterator can be in arbitrary order (sort+unique is applied).
        void load_from_iterator(size_t _rows, size_t _cols, size_t _nnz, entry_iterator_t<val_type>* entry_it) {
            clear_space(); // clear any pre-allocated space in case of memory leak
            rows =_rows, cols=_cols, nnz=_nnz;
            allocate_space(rows,cols,nnz);

            // a trick to utilize the space that have been allocated
            std::vector<size_t> perm(nnz);
            unsigned *tmp_row_idx = col_idx;
            unsigned *tmp_col_idx = row_idx;
            val_type *tmp_val = val;

            for(size_t idx = 0; idx < nnz; idx++){
                entry_t<val_type> rate = entry_it->next();

                tmp_row_idx[idx] = rate.i;
                tmp_col_idx[idx] = rate.j;
                tmp_val[idx] = rate.v;

                perm[idx] = idx;
            }

            // TODO can change to O(n) method
            // sort entries into row-majored ordering
            std::sort(perm.begin(), perm.end(), SparseLess(tmp_row_idx, tmp_col_idx));

            // add up the values in the same position (i, j)
            size_t cur_nnz = 0;
            for(size_t idx = 0; idx < nnz; idx++) {
                if(cur_nnz > 0
                && tmp_row_idx[perm[idx]] == tmp_row_idx[perm[cur_nnz-1]]
                && tmp_col_idx[perm[idx]] == tmp_col_idx[perm[cur_nnz-1]])
                    tmp_val[perm[cur_nnz-1]] += tmp_val[perm[idx]];
                else {
                    tmp_row_idx[perm[cur_nnz]] = tmp_row_idx[perm[idx]];
                    tmp_col_idx[perm[cur_nnz]] = tmp_col_idx[perm[idx]];
                    tmp_val[perm[cur_nnz]] = tmp_val[perm[idx]];
                    cur_nnz ++;
                }
            }
            nnz = cur_nnz;

            for(size_t idx = 0; idx < nnz; idx++){
                row_ptr[tmp_row_idx[perm[idx]] + 1] ++;
                col_ptr[tmp_col_idx[perm[idx]] + 1] ++;
            }

            // Generate CSR format
            for(size_t idx = 0; idx < nnz; idx++) {
                val_t[idx] = tmp_val[perm[idx]];
                col_idx[idx] = tmp_col_idx[perm[idx]];
            }

            // Calculate nnz for each row and col
            max_row_nnz = max_col_nnz = 0;
            for(size_t r = 1; r <= rows; r++) {
                max_row_nnz = std::max(max_row_nnz, row_ptr[r]);
                row_ptr[r] += row_ptr[r-1];
            }
            for(size_t c = 1; c <= cols; c++) {
                max_col_nnz = std::max(max_col_nnz, col_ptr[c]);
                col_ptr[c] += col_ptr[c-1];
            }

            // Transpose CSR into CSC matrix
            for(size_t r = 0; r < rows; r++){
                for(size_t idx = row_ptr[r]; idx < row_ptr[r+1]; idx++){
                    size_t c = (size_t) col_idx[idx];
                    row_idx[col_ptr[c]] = r;
                    val[col_ptr[c]++] = val_t[idx];
                }
            }

            for(size_t c = cols; c > 0; c--) col_ptr[c] = col_ptr[c-1];
            col_ptr[0] = 0;

        }

        void load(size_t _rows, size_t _cols, size_t _nnz, const char *filename, format_t fmt) {
            if(fmt == smat_t<val_type>::TXT) {
                file_iterator_t<val_type> entry_it(_nnz, filename);
                load_from_iterator(_rows, _cols, _nnz, &entry_it);
            } else if(fmt == smat_t<val_type>::PETSc) {
                load_from_PETSc(filename);
            } else if(fmt == smat_t<val_type>::SVMLIGHT) {
                load_from_svmlight(filename);
            } else {
                fprintf(stderr, "Error: filetype %d not supported\n", fmt);
                return;
            }
        }

        void load_from_svmlight(const char *filename, size_t nr_skips=1, bool zero_based=false, double append_bias=-1.0) {
            svmlight_file_iterator_t<val_type> entry_it(filename, nr_skips, zero_based, append_bias);
            load_from_iterator(entry_it.get_rows(), entry_it.get_cols(), entry_it.get_nnz(), &entry_it);
        }

        void load_from_PETSc(const char *filename) {
            FILE *fp = fopen(filename, "rb");
            if(fp == NULL) {
                fprintf(stderr, "Error: can't read the file (%s)!!\n", filename);
                return;
            }
            load_from_PETSc(fp, filename);
            fclose(fp);
        }
        void load_from_PETSc(FILE *fp, const char *filename=NULL) {
            clear_space(); // clear any pre-allocated space in case of memory leak
            const int UNSIGNED_FILE = 1211216, LONG_FILE = 1015;
            int32_t int_buf[3];
            size_t headersize = 0;
            headersize += sizeof(int)*fread(int_buf, sizeof(int), 3, fp);
            int filetype = int_buf[0];
            rows = (size_t) int_buf[1];
            cols = (size_t) int_buf[2];
            if(filetype == UNSIGNED_FILE) {
                headersize += sizeof(int)*fread(int_buf, sizeof(int32_t), 1, fp);
                nnz = (size_t) int_buf[0];
            } else if (filetype == LONG_FILE){
                headersize += sizeof(size_t)*fread(&nnz, sizeof(int64_t), 1, fp);
            } else {
                fprintf(stderr, "Error: wrong PETSc format in %s.\n", filename);
            }
            allocate_space(rows,cols,nnz);
            // load CSR from the binary PETSc format
            {
                // read row_ptr
                std::vector<int32_t> nnz_row(rows);
                headersize += sizeof(int32_t)*fread(&nnz_row[0], sizeof(int32_t), rows, fp);
                row_ptr[0] = 0;
                for(size_t r = 1; r <= rows; r++)
                    row_ptr[r] = row_ptr[r-1] + nnz_row[r-1];
                // read col_idx
                headersize += sizeof(int)*fread(&col_idx[0], sizeof(unsigned), nnz, fp);

                // read val_t
                const size_t chunksize = 1024;
                double buf[chunksize];
                size_t idx = 0;
                while(idx + chunksize < nnz) {
                    headersize += sizeof(double)*fread(&buf[0], sizeof(double), chunksize, fp);
                    for(size_t i = 0; i < chunksize; i++)
                        val_t[idx+i] = (val_type) buf[i];
                    idx += chunksize;
                }
                size_t remaining = nnz - idx;
                headersize += sizeof(double)*fread(&buf[0], sizeof(double), remaining, fp);
                for(size_t i = 0; i < remaining; i++)
                    val_t[idx+i] = (val_type) buf[i];
            }

            csr_to_csc();
            update_max_nnz();
        }

        void save_PETSc_to_file(const char *filename) const {
            FILE *fp = fopen(filename, "wb");
            if(fp == NULL) {
                fprintf(stderr,"Error: can't open file %s\n", filename);
                exit(1);
            }
            save_PETSc_to_file(fp);
        }
        void save_PETSc_to_file(FILE *fp) const {
            const int UNSIGNED_FILE = 1211216, LONG_FILE = 1015;
            int32_t int_buf[3] = {(int32_t)LONG_FILE, (int32_t)rows, (int32_t)cols};
            std::vector<int32_t> nnz_row(rows);
            for(size_t r = 0; r < rows; r++)
                nnz_row[r] = (int)nnz_of_row(r);

            fwrite(&int_buf[0], sizeof(int32_t), 3, fp);
            fwrite(&nnz, sizeof(size_t), 1, fp);
            fwrite(&nnz_row[0], sizeof(int32_t), rows, fp);
            fwrite(&col_idx[0], sizeof(unsigned), nnz, fp);

            // the following part == fwrite(val_t, sizeof(double), nnz, fp);
            const size_t chunksize = 1024;
            double buf[chunksize];
            size_t idx = 0;
            while(idx + chunksize < nnz) {
                for(size_t i = 0; i < chunksize; i++)
                    buf[i] = (double) val_t[idx+i];
                fwrite(&buf[0], sizeof(double), chunksize, fp);
                idx += chunksize;
            }
            size_t remaining = nnz - idx;
            for(size_t i = 0; i < remaining; i++)
                buf[i] = (double) val_t[idx+i];
            fwrite(&buf[0], sizeof(double), remaining, fp);
        }

        val_type get_global_mean() const {
            val_type sum=0;
            for(size_t idx = 0; idx < nnz; idx++) sum += val[idx];
            return sum / (val_type)nnz;
        }
        void remove_bias(val_type bias=0) {
            if(bias) {
                for(size_t idx = 0; idx < nnz; idx++) {
                    val[idx] -= bias;
                    val_t[idx] -= bias;
                }
            }
        }

        void print_mat(const char *str="", FILE *fp=stdout) const {
            fprintf(fp, "===>%s<===\n", str);
            fprintf(fp, "rows %lu, cols %lu, nnz %lu\n", rows, cols, nnz);
            fprintf(fp, "col_ptr, row_idx, val = %p, %p, %p\n", col_ptr, row_idx, val);
            fprintf(fp, "row_ptr, col_idx, val_t = %p, %p, %p\n", row_ptr, col_idx, val_t);
            fprintf(fp, "mem_alloc_by_me = %d\n", mem_alloc_by_me);

            fprintf(fp, "Matrix:\n");
            for(size_t i = 0; i < rows; i++) {
                size_t it = row_ptr[i];
                for(size_t j = 0; j < cols; j++) {
                    if(it < row_ptr[i+1] && col_idx[it] == j) {
                        fprintf(fp, "%.3f ", val_t[it]);
                        it ++;
                    }
                    else
                        fprintf(fp, "0.000 ");
                }
                fprintf(fp, "\n");
            }

            fprintf(fp, "Matrix^T:\n");
            for(size_t i = 0; i < cols; i++) {
                size_t it = col_ptr[i];
                for(size_t j = 0; j < rows; j++) {
                    if(it < col_ptr[i+1] && row_idx[it] == j) {
                        fprintf(fp, "%.3f ", val[it]);
                        it ++;
                    }
                    else
                        fprintf(fp, "0.000 ");
                }
                fprintf(fp, "\n");
            }
        }

        // ===========================================
        // ========= Friend Functions/Classes ========
        // ===========================================
        template<typename VX, typename VY, typename VZ>
        friend smat_t<VZ>& smat_x_smat(const smat_t<VX> &X, const smat_t<VY> &Y, smat_t<VZ> &Z, int threads);

        template<typename VX, typename VY, typename VZ>
        friend smat_t<VZ>& smat_x_smat_single_thread(const smat_t<VX> &X, const smat_t<VY> &Y, smat_t<VZ> &Z);

};


#ifdef __cplusplus
extern "C" {
#endif

// rows, cols, nnz, &row_ptr, &col_ptr, &val_ptr
typedef void(*py_coo_allocator_t)(uint64_t, uint64_t, uint64_t, void*, void*, void*);




#ifdef __cplusplus
} // extern
#endif

template<typename val_type>
struct coo_t {
    size_t rows;
    size_t cols;
    std::vector<size_t> row_idx;
    std::vector<size_t> col_idx;
    std::vector<val_type> val;

    coo_t(size_t rows=0, size_t cols=0): rows(rows), cols(cols) {}

    size_t nnz() const { return val.size(); }

    void reshape(size_t rows_, size_t cols_) {
        rows = rows_;
        cols = cols_;
        clear();
    }

    void clear() {
        row_idx.clear();
        col_idx.clear();
        val.clear();
    }

    void reserve(size_t capacity) {
        row_idx.reserve(capacity);
        col_idx.reserve(capacity);
        val.reserve(capacity);
    }

    void swap(coo_t& other) {
        std::swap(rows, other.rows);
        std::swap(cols, other.cols);
        row_idx.swap(other.row_idx);
        col_idx.swap(other.col_idx);
        val.swap(other.val);
    }

    void extends(coo_t& other) {
        std::copy(other.row_idx.begin(), other.row_idx.end(), std::back_inserter(row_idx));
        std::copy(other.col_idx.begin(), other.col_idx.end(), std::back_inserter(col_idx));
        std::copy(other.val.begin(), other.val.end(), std::back_inserter(val));
    }

    template<typename I, typename V>
    void push_back(I i, I j, V x, double threshold=0) {
        if(fabs(x) >= threshold) {
            row_idx.push_back(i);
            col_idx.push_back(j);
            val.push_back(x);
        }
    }

    void create_smat(smat_t<val_type>& X) {
        coo_iterator_t<val_type> it(nnz(), row_idx.data(), col_idx.data(), val.data());
        X.load_from_iterator(rows, cols, nnz(), &it);
    }

    void create_pycoo(const py_coo_allocator_t& alloc) const {
        uint64_t* row_ptr=NULL;
        uint64_t* col_ptr=NULL;
        val_type* val_ptr=NULL;
        alloc(rows, cols, nnz(), &row_ptr, &col_ptr, &val_ptr);
        for(size_t i = 0; i < nnz(); i++) {
            row_ptr[i] = row_idx[i];
            col_ptr[i] = col_idx[i];
            val_ptr[i] = val[i];
        }
    }
};

/*-------------- Iterators -------------------*/

template<typename val_type>
class entry_t{
    public:
        unsigned i, j;
        val_type v, weight;
        entry_t(int _i=0, int _j=0, val_type _v=0, val_type _w=1.0): i(_i), j(_j), v(_v), weight(_w){}
};

template<typename val_type>
class entry_iterator_t {
    public:
        // Number of elements left to iterate
        size_t nnz;

        // When no next entry, return entry_t(0, 0, 0, -1);
        virtual entry_t<val_type> next() = 0;

        size_t get_nnz() const { return nnz; }
};

#define MAXLINE 10240
// Iterator for files with (i,j,v) tuples
template<typename val_type>
class file_iterator_t: public entry_iterator_t<val_type> {
    public:
        using entry_iterator_t<val_type>::nnz;

        file_iterator_t(size_t nnz_, const char* filename, size_t start_pos=0) {
            nnz = nnz_;
            fp = fopen(filename,"rb");
            if(fp == NULL) {
                fprintf(stderr, "Error: cannot read the file (%s)!!\n", filename);
                return;
            }
            fseek(fp, start_pos, SEEK_SET);
        }

        ~file_iterator_t(){ if (fp) fclose(fp); }

        entry_t<val_type> next() {
            const int base10 = 10;
            if(nnz > 0) {
                --nnz;
                if(fgets(&line[0], MAXLINE, fp)==NULL)
                    fprintf(stderr, "Error: reading error !!\n");
                char *head_ptr = &line[0];
                size_t i = strtol(head_ptr, &head_ptr, base10);
                size_t j = strtol(head_ptr, &head_ptr, base10);
                double v = strtod(head_ptr, &head_ptr);
                return entry_t<val_type>(i - 1, j - 1, (val_type)v);
            }
            else { // No more to iterate
                return entry_t<val_type>(0, 0, 0, -1);
            }
        }

    private:
        FILE *fp;
        char line[MAXLINE];
};

template<class val_type>
class svmlight_file_iterator_t : public entry_iterator_t<val_type> {
    public:
        using entry_iterator_t<val_type>::nnz;

        svmlight_file_iterator_t(
                const char* filename,
                size_t nr_skips=1,
                bool zero_based=false,
                double append_bias=-1.0) {

            std::ifstream fs;
            std::string line, kv;
            const int base10 = 10;

            fs.open(filename, std::ios::in);
            if(!fs.is_open()) {
                std::cout << "Unable to open" << filename << std::endl;
                exit(-1);
            }

            I.clear();
            J.clear();
            V.clear();
            nr_rows = nr_cols = 0;

            while(std::getline(fs, line)) {
                if(fs.eof()) {
                    break;
                }
                std::stringstream line_ss;
                line_ss.str(line);
                if(nr_skips != 0) {
                    // skip label part;
                    for(size_t i = 0; i < nr_skips; i++) {
                        line_ss >> kv;
                    }
                }
                size_t row_idx = nr_rows;
                while(line_ss >> kv) {
                    char *head_ptr = const_cast<char*>(kv.c_str());
                    size_t key = strtol(head_ptr, &head_ptr, base10);
                    head_ptr++;  // advancing for the ":" seperator
                    val_type val = static_cast<val_type>(strtod(head_ptr, &head_ptr));
                    size_t col_idx = (zero_based)? key : (key - 1);
                    nr_cols = std::max(nr_cols, col_idx + 1);

                    I.push_back(row_idx);
                    J.push_back(col_idx);
                    V.push_back(val);
                }
                nr_rows += 1;
            }
            if(append_bias > 0) {
                size_t col_idx = nr_cols;
                nr_cols += 1;
                val_type val = static_cast<val_type>(append_bias);

                for(size_t row_idx = 0; row_idx < nr_rows; row_idx++) {
                    I.push_back(row_idx);
                    J.push_back(col_idx);
                    V.push_back(val);
                }
            }
            idx = 0;
            nnz = I.size();
        }

        entry_t<val_type> next() {
            if(nnz > 0) {
                ++idx; --nnz;
                return entry_t<val_type>(I[idx - 1], J[idx - 1], V[idx - 1]);
            } else {
                return entry_t<val_type>(0, 0, 0, -1);
            }
        }

        size_t get_rows() const { return nr_rows; }
        size_t get_cols() const { return nr_cols; }

    private:
        size_t nr_rows, nr_cols;
        size_t idx;
        std::vector<size_t> I, J;
        std::vector<val_type> V;
};

// Iterator for three vectors (I, J, V)
template<typename val_type>
class coo_iterator_t: public entry_iterator_t<val_type> {
    public:
        using entry_iterator_t<val_type>::nnz;

        coo_iterator_t(const std::vector<size_t> _I, const std::vector<size_t> _J, const std::vector<val_type> _V){
            nnz = std::min(std::min(_I.size(), _J.size()), _V.size());

            idx = 0;
            I = &_I[0]; J = &_J[0]; V = &_V[0];
        }

        coo_iterator_t(const size_t _nnz, const size_t* _I, const size_t* _J, const val_type* _V){
            nnz = _nnz;

            idx = 0;
            I = _I; J = _J; V = _V;
        }

        ~coo_iterator_t(){ }

        entry_t<val_type> next() {
            if(nnz > 0) {
                ++idx;
                --nnz;
                return entry_t<val_type>(I[idx - 1], J[idx - 1], V[idx - 1]);
            } else {
                return entry_t<val_type>(0, 0, 0, -1);
            }
        }

    private:
        int idx;

        const size_t *I, *J;
        const val_type *V;
};

// Iterator for sparse matrix
template<typename val_type>
class smat_iterator_t: public entry_iterator_t<val_type> {
    public:
        using entry_iterator_t<val_type>::nnz;

        smat_iterator_t(const smat_t<val_type>& M, major_t major = ROWMAJOR) {
            nnz = M.nnz;
            col_idx = (major == ROWMAJOR)? M.col_idx: M.row_idx;
            row_ptr = (major == ROWMAJOR)? M.row_ptr: M.col_ptr;
            val_t = (major == ROWMAJOR)? M.val_t: M.val;
            rows = (major==ROWMAJOR)? M.rows: M.cols;
            cols = (major==ROWMAJOR)? M.cols: M.rows;
            cur_idx = cur_row = 0;
        }

        ~smat_iterator_t() {}

        entry_t<val_type> next() {
            if (nnz > 0)
                nnz--;
            else
                return entry_t<val_type>(0, 0, 0, -1);

            while (cur_idx >= row_ptr[cur_row+1])
                cur_row++;

            entry_t<val_type> ret(cur_row, col_idx[cur_idx], val_t[cur_idx]);
            cur_idx++;
            return ret;
        }

    private:
        unsigned *col_idx;
        size_t *row_ptr;
        val_type *val_t;
        size_t rows, cols, cur_idx;
        size_t cur_row;
};

// Iterator for a subset of sparse matrix
template<typename val_type>
class smat_subset_iterator_t: public entry_iterator_t<val_type> {
    public:
        using entry_iterator_t<val_type>::nnz;

        // When ROWMAJOR (COLMAJOR) is used, we sample several rows (columns) according to the order in subset_.
        // If remapping = true, then we are using the corresponding index (i, j) in the submatrix.
        // If remapping = false, then we are using the index (i, j) in the original matrix.
        smat_subset_iterator_t(const smat_t<val_type>& M, const unsigned *subset_, size_t size, major_t major_ = ROWMAJOR, bool remapping_=true) {
            major = major_; remapping = remapping_;

            cr_idx = (major == ROWMAJOR)? M.col_idx: M.row_idx;
            rc_ptr = (major == ROWMAJOR)? M.row_ptr: M.col_ptr;
            val_t = (major == ROWMAJOR)? M.val_t: M.val;

            rows = (major==ROWMAJOR)? (remapping? size: M.rows): M.rows;
            cols = (major==ROWMAJOR)? M.cols: (remapping? size: M.cols);
            subset.resize(size);

            nnz = 0;
            for(size_t i = 0; i < size; i++) {
                unsigned idx = subset_[i];
                subset[i] = idx;
                nnz += (major == ROWMAJOR)? M.nnz_of_row(idx): M.nnz_of_col(idx);
            }

            cur_rc = 0;
            cur_idx = rc_ptr[subset[cur_rc]];
        }

        smat_subset_iterator_t(const smat_t<val_type>& M, const std::vector<unsigned> &subset_, major_t major_ = ROWMAJOR, bool remapping_=true) {
            major = major_; remapping = remapping_;

            cr_idx = (major == ROWMAJOR)? M.col_idx: M.row_idx;
            rc_ptr = (major == ROWMAJOR)? M.row_ptr: M.col_ptr;
            val_t = (major == ROWMAJOR)? M.val_t: M.val;

            rows = (major==ROWMAJOR)? (remapping? subset_.size(): M.rows): M.rows;
            cols = (major==ROWMAJOR)? M.cols: (remapping? subset_.size(): M.cols);
            subset.resize(subset_.size());

            nnz = 0;
            for(size_t i = 0; i < subset_.size(); i++) {
                unsigned idx = subset_[i];
                subset[i] = idx;
                nnz += (major == ROWMAJOR)? M.nnz_of_row(idx): M.nnz_of_col(idx);
            }

            cur_rc = 0;
            cur_idx = rc_ptr[subset[cur_rc]];
        }

        ~smat_subset_iterator_t() {}

        size_t get_rows() { return rows; }
        size_t get_cols() { return cols; }

        entry_t<val_type> next() {
            if (nnz > 0)
                nnz--;
            else
                return entry_t<val_type>(0,0,0, -1);

            while (cur_idx >= rc_ptr[subset[cur_rc]+1]) {
                cur_rc++;
                cur_idx = rc_ptr[subset[cur_rc]];
            }

            entry_t<val_type> ret_rowwise(remapping? cur_rc: subset[cur_rc], cr_idx[cur_idx], val_t[cur_idx]);
            entry_t<val_type> ret_colwise(cr_idx[cur_idx], remapping? cur_rc: subset[cur_rc], val_t[cur_idx]);

            cur_idx++;
            return major==ROWMAJOR? ret_rowwise: ret_colwise;
        }

    private:
        size_t rows, cols;
        std::vector<unsigned>subset;

        unsigned *cr_idx;
        size_t *rc_ptr;
        val_type *val_t;

        size_t cur_rc, cur_idx;

        major_t major;
        bool remapping;
};

// Iterator for a dense matrix
template<typename val_type>
class dmat_iterator_t: public entry_iterator_t<val_type> {
    public:
        using entry_iterator_t<val_type>::nnz;

        dmat_iterator_t(const dmat_t<val_type>& M, double threshold=1e-12) : M(M), rows(M.rows), cols(M.cols), threshold(fabs(threshold)) {
            cur_row = 0;
            cur_col = 0;
            nnz = 0;

            bool find_firstnz = true;
            for(size_t i = 0; i < rows; i++)
                for(size_t j = 0; j < cols; j++)
                    if(fabs((double)M.at(i,j)) >= threshold) {
                        if(find_firstnz) {
                            cur_row = i;
                            cur_col = j;
                            find_firstnz = false;
                        }
                        nnz++;
                    }
        }

        ~dmat_iterator_t() {}

        entry_t<val_type> next() {
            if (nnz > 0)
                nnz--;
            else
                return entry_t<val_type>(0,0,0, -1);

            entry_t<val_type> entry(cur_row, cur_col, M.at(cur_row, cur_col));

            do {
                cur_col ++;
                if(cur_col == cols) {
                    cur_row ++;
                    cur_col = 0;
                }
            } while(fabs((double)M.at(cur_row, cur_col)) < threshold);

            return entry;
        }

    private:
        const dmat_t<val_type>& M;
        size_t rows, cols, cur_row, cur_col;
        double threshold;
};

/*-------------- Implementation of Linear Algebra Operations --------------*/

// Lapack and Blas support
#ifdef _WIN32
#define ddot_ ddot
#define sdot_ sdot
#define daxpy_ daxpy
#define saxpy_ saxpy
#define dcopy_ dcopy
#define scopy_ scopy
#define dgemm_ dgemm
#define sgemm_ sgemm
#define dposv_ dposv
#define sposv_ sposv
#define dgesdd_ dgesdd
#define sgesdd_ sgesdd
#endif

extern "C" {

    double ddot_(ptrdiff_t *, double *, ptrdiff_t *, double *, ptrdiff_t *);
    float sdot_(ptrdiff_t *, float *, ptrdiff_t *, float *, ptrdiff_t *);

    ptrdiff_t dscal_(ptrdiff_t *, double *, double *, ptrdiff_t *);
    ptrdiff_t sscal_(ptrdiff_t *, float *, float *, ptrdiff_t *);

    ptrdiff_t daxpy_(ptrdiff_t *, double *, double *, ptrdiff_t *, double *, ptrdiff_t *);
    ptrdiff_t saxpy_(ptrdiff_t *, float *, float *, ptrdiff_t *, float *, ptrdiff_t *);

    double dcopy_(ptrdiff_t *, double *, ptrdiff_t *, double *, ptrdiff_t *);
    float scopy_(ptrdiff_t *, float *, ptrdiff_t *, float *, ptrdiff_t *);

    void dgemm_(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, double *alpha, double *a, ptrdiff_t *lda, double *b, ptrdiff_t *ldb, double *beta, double *c, ptrdiff_t *ldc);
    void sgemm_(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, float *alpha, float *a, ptrdiff_t *lda, float *b, ptrdiff_t *ldb, float *beta, float *c, ptrdiff_t *ldc);

    int dposv_(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, double *a, ptrdiff_t *lda, double *b, ptrdiff_t *ldb, ptrdiff_t *info);
    int sposv_(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, float *a, ptrdiff_t *lda, float *b, ptrdiff_t *ldb, ptrdiff_t *info);

    void dgesdd_(char* jobz, ptrdiff_t* m, ptrdiff_t* n, double* a, ptrdiff_t* lda, double* s, double* u, ptrdiff_t* ldu, double* vt, ptrdiff_t* ldvt, double* work, ptrdiff_t* lwork, ptrdiff_t* iwork, ptrdiff_t* info);
    void sgesdd_(char* jobz, ptrdiff_t* m, ptrdiff_t* n, float* a, ptrdiff_t* lda, float* s, float* u, ptrdiff_t* ldu, float* vt, ptrdiff_t* ldvt, float* work, ptrdiff_t* lwork, ptrdiff_t* iwork, ptrdiff_t* info);

}

template<typename val_type> val_type dot(ptrdiff_t *, val_type *, ptrdiff_t *, val_type *, ptrdiff_t *);
template<> inline double dot(ptrdiff_t *len, double *x, ptrdiff_t *xinc, double *y, ptrdiff_t *yinc) { return ddot_(len,x,xinc,y,yinc);}
template<> inline float dot(ptrdiff_t *len, float *x, ptrdiff_t *xinc, float *y, ptrdiff_t *yinc) { return sdot_(len,x,xinc,y,yinc);}

template<typename val_type> val_type scal(ptrdiff_t *, val_type *, val_type *, ptrdiff_t *);
template<> inline double scal(ptrdiff_t *len, double *a, double *x, ptrdiff_t *xinc) { return dscal_(len,a,x,xinc);}
template<> inline float scal(ptrdiff_t *len, float *a,  float *x, ptrdiff_t *xinc) { return sscal_(len,a,x,xinc);}

template<typename val_type> ptrdiff_t axpy(ptrdiff_t *, val_type *, val_type *, ptrdiff_t *, val_type *, ptrdiff_t *);
template<> inline ptrdiff_t axpy(ptrdiff_t *len, double *alpha, double *x, ptrdiff_t *xinc, double *y, ptrdiff_t *yinc) { return daxpy_(len,alpha,x,xinc,y,yinc);};
template<> inline ptrdiff_t axpy(ptrdiff_t *len, float *alpha, float *x, ptrdiff_t *xinc, float *y, ptrdiff_t *yinc) { return saxpy_(len,alpha,x,xinc,y,yinc);};

template<typename val_type> val_type copy(ptrdiff_t *, val_type *, ptrdiff_t *, val_type *, ptrdiff_t *);
template<> inline double copy(ptrdiff_t *len, double *x, ptrdiff_t *xinc, double *y, ptrdiff_t *yinc) { return dcopy_(len,x,xinc,y,yinc);}
template<> inline float copy(ptrdiff_t *len, float *x, ptrdiff_t *xinc, float *y, ptrdiff_t *yinc) { return scopy_(len,x,xinc,y,yinc);}

template<typename val_type> void gemm(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, val_type *alpha, val_type *a, ptrdiff_t *lda, val_type *b, ptrdiff_t *ldb, val_type *beta, val_type *c, ptrdiff_t *ldc);
template<> inline void gemm(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, double *alpha, double *a, ptrdiff_t *lda, double *b, ptrdiff_t *ldb, double *beta, double *c, ptrdiff_t *ldc) { dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
template<> inline void gemm<float>(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, float *alpha, float *a, ptrdiff_t *lda, float *b, ptrdiff_t *ldb, float *beta, float *c, ptrdiff_t *ldc) { sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }

template<typename val_type> int posv(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, val_type *a, ptrdiff_t *lda, val_type *b, ptrdiff_t *ldb, ptrdiff_t *info);
template<> inline int posv(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, double *a, ptrdiff_t *lda, double *b, ptrdiff_t *ldb, ptrdiff_t *info) { return dposv_(uplo, n, nrhs, a, lda, b, ldb, info); }
template<> inline int posv(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, float *a, ptrdiff_t *lda, float *b, ptrdiff_t *ldb, ptrdiff_t *info) { return sposv_(uplo, n, nrhs, a, lda, b, ldb, info); }

template<typename val_type> void gesdd(char* jobz, ptrdiff_t* m, ptrdiff_t* n, val_type* a, ptrdiff_t* lda, val_type* s, val_type* u, ptrdiff_t* ldu, val_type* vt, ptrdiff_t* ldvt, val_type* work, ptrdiff_t* lwork, ptrdiff_t* iwork, ptrdiff_t* info);
template<> inline void gesdd(char* jobz, ptrdiff_t* m, ptrdiff_t* n, double* a, ptrdiff_t* lda, double* s, double* u, ptrdiff_t* ldu, double* vt, ptrdiff_t* ldvt, double* work, ptrdiff_t* lwork, ptrdiff_t* iwork, ptrdiff_t* info) { return dgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info); }
template<> inline void gesdd(char* jobz, ptrdiff_t* m, ptrdiff_t* n, float* a, ptrdiff_t* lda, float* s, float* u, ptrdiff_t* ldu, float* vt, ptrdiff_t* ldvt, float* work, ptrdiff_t* lwork, ptrdiff_t* iwork, ptrdiff_t* info) { return sgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info); }


// <x,y>
template<typename val_type>
val_type do_dot_product(const val_type *x, const val_type *y, size_t size) {
    val_type *xx = const_cast<val_type*>(x);
    val_type *yy = const_cast<val_type*>(y);
    ptrdiff_t inc = 1;
    ptrdiff_t len = (ptrdiff_t) size;
    return dot(&len, xx, &inc, yy, &inc);
}
template<typename val_type>
val_type do_dot_product(const dvec_t<val_type> &x, const dvec_t<val_type> &y) {
    assert(x.size() == y.size());
    return do_dot_product(x.data(), y.data(), x.size());
}

template<typename val_type>
val_type do_dot_product(const svec_t<val_type> &x, const svec_t<val_type> &y) {
    if(x.nnz > y.nnz) { return do_dot_product(y, x); }

    val_type ret = 0;
    size_t s = 0, t = 0;
    unsigned *xend = x.idx + x.nnz;
    unsigned *yend = y.idx + y.nnz;
    while(s < x.nnz && t < y.nnz) {
        if(x.idx[s] == y.idx[t]) {
            ret += x.val[s] * y.val[t];
            s++;
            t++;
        } else if(x.idx[s] < y.idx[t]) {
            s = std::lower_bound(x.idx + s, xend, y.idx[t]) - x.idx;
        } else {
            t = std::lower_bound(y.idx + t, yend, x.idx[s]) - y.idx;
        }
    }
    return ret;
}
template<typename val_type>
val_type do_dot_product_old(const svec_t<val_type> &x, const svec_t<val_type> &y) {
    assert(x.size() == y.size());
    val_type ret = 0;
    for(size_t i = 0, j = 0; i < x.get_nnz() && j < y.get_nnz();) {
        if(x.idx[i] < y.idx[j]) {
            i ++;
        } else if(x.idx[i] > y.idx[j]) {
            j ++;
        } else {
            ret += x.val[i] * y.val[j];
            i ++; j ++;
        }
    }
    return ret;
}
template<typename val_type>
val_type do_dot_product(const sdvec_t<val_type> &x, const sdvec_t<val_type> &y) {
    assert(x.size() == y.size());
    val_type ret = 0;
    for(size_t i = 0, j = 0; i < x.get_nnz() && j < y.get_nnz();) {
        if(x.nz_idx[i] < y.nz_idx[j]) {
            i++;
        } else if(x.nz_idx[i] < y.nz_idx[i]) {
            j++;
        } else {
            ret += x[x.nz_idx[i]] * y[y.nz_idx[j]];
            i++;
            j++;
        }
    }
    return ret;
}
template<typename val_type>
val_type do_dot_product(const dvec_t<val_type> &x, const svec_t<val_type> &y) {
    assert(x.size() == y.size());
    val_type ret = 0;
    for(size_t i = 0; i < y.get_nnz(); i++)
        ret += x[y.idx[i]] * y.val[i];
    return ret;
}
template<typename val_type>
val_type do_dot_product(const svec_t<val_type> &x, const dvec_t<val_type> &y) {
    assert(x.size() == y.size());
    return do_dot_product(y, x);
}
template<typename val_type>
val_type do_dot_product(const dvec_t<val_type> &x, const sdvec_t<val_type> &y) {
    val_type ret = 0;
    for(size_t i = 0; i < y.get_nnz(); i++) {
        ret += x[y.nz_idx[i]] * y[y.nz_idx[i]];
    }
    return ret;
}
template<typename val_type>
val_type do_dot_product(const sdvec_t<val_type> &x, const dvec_t<val_type> &y) {
    return do_dot_product(y, x);
}
template<typename val_type>
val_type do_dot_product_old(const svec_t<val_type> &x, const sdvec_t<val_type> &y) {
    val_type ret = 0;
    for(size_t i = 0, j = 0; i < x.get_nnz() && j < y.get_nnz();) {
        if(x.idx[i] < y.nz_idx[j]) {
            i++;
        } else if(x.idx[i] > y.nz_idx[j]) {
            j++;
        } else {
            ret += x.val[i] * y[y.nz_idx[j]];
            i++;
            j++;
        }
    }
    return ret;
}
template<typename val_type>
val_type do_dot_product(const sdvec_t<val_type> &x, const svec_t<val_type> &y) {
    return do_dot_product(y, x);
}
template<typename val_type>
val_type do_dot_product(const gvec_t<val_type> &x, const gvec_t<val_type> &y) {
    assert(x.size() == y.size());
    if(x.is_sparse() && y.is_sparse())
        return do_dot_product(x.get_sparse(), y.get_sparse());
    else if(x.is_sparse() && y.is_dense())
        return do_dot_product(x.get_sparse(), y.get_dense());
    else if(x.is_dense() && y.is_sparse())
        return do_dot_product(x.get_dense(), y.get_sparse());
    else if(x.is_dense() && y.is_dense())
        return do_dot_product(x.get_dense(), y.get_dense());
    else
        return 0;
}
template<typename val_type>
val_type do_dot_product(const dmat_t<val_type> &x, const dmat_t<val_type> &y) {
    assert(x.rows == y.rows && x.cols == y.cols);
    if((x.is_rowmajor() && y.is_rowmajor()) || (x.is_colmajor() && y.is_colmajor()))
        return do_dot_product(x.data(), y.data(), x.rows*x.cols);
    else {
        val_type ret = 0.0;
        const dmat_t<val_type> &xx = (x.rows > x.cols) ? x : x.transpose();
        const dmat_t<val_type> &yy = (y.rows > y.cols) ? y : y.transpose();
#pragma omp parallel for schedule(static) reduction(+:ret)
        for(size_t i = 0; i < xx.rows; i++) {
            double ret_local = 0.0;
            for(size_t j = 0; j < xx.cols; j++)
                ret_local += xx.at(i,j)*yy.at(i,j);
            ret += ret_local;
        }
        return (val_type)ret;
    }
}
template<typename val_type>
val_type do_dot_product(const smat_t<val_type> &x, const smat_t<val_type> &y) {
    assert(x.rows == y.rows && x.cols == y.cols);
    val_type ret = 0.0;
    const smat_t<val_type> &xx = (x.rows > x.cols) ? x : x.transpose();
    const smat_t<val_type> &yy = (y.rows > y.cols) ? y : y.transpose();
#pragma omp parallel for schedule(static) reduction(+:ret)
    for(size_t i = 0; i < xx.rows; i++) {
        svec_t<val_type> sv1 = xx.get_row(i);
        svec_t<val_type> sv2 = yy.get_row(i);
        val_type ret_local = do_dot_product(sv1, sv2);
        ret += ret_local;
    }
    return (val_type)ret;
}
template<typename val_type>
val_type do_dot_product(const smat_t<val_type> &x, const dmat_t<val_type>&y) {
    assert(x.rows == y.rows && x.cols == y.cols);
    double ret = 0;
    const smat_t<val_type> &xx = (x.rows > x.cols) ? x : x.transpose();
#pragma omp parallel for schedule(static) reduction(+:ret)
    for(size_t i = 0; i < xx.rows; i++) {
        double tmp = 0;
        for(size_t idx = xx.row_ptr[i]; idx < xx.row_ptr[i + 1]; idx++) {
            tmp += xx.val[idx] * y.at(i, xx.col_idx[idx]);
        }
        ret += tmp;
    }
    return static_cast<val_type>(ret);
}
template<typename val_type>
val_type do_dot_product(const dmat_t<val_type>&x, const smat_t<val_type> &y) {
    return do_dot_product(y, x);
}
template<typename val_type>
val_type do_dot_product(const gmat_t<val_type>&x, const gmat_t<val_type> &y) {
    assert(x.rows == y.rows && x.cols == y.cols);
    if(x.is_sparse() && y.is_sparse())
        return do_dot_product(x.get_sparse(), y.get_sparse());
    else if(x.is_sparse() && y.is_dense())
        return do_dot_product(x.get_sparse(), y.get_dense());
    else if(x.is_dense() && y.is_sparse())
        return do_dot_product(x.get_dense(), y.get_sparse());
    else if(x.is_dense() && y.is_dense())
        return do_dot_product(x.get_dense(), y.get_dense());
    else
        return 0;
}

// y = alpha * x + y
template<typename val_type, typename T>
val_type* do_axpy(T alpha, const val_type *x, val_type *y, size_t size) {
    if(alpha == 0) return y;
    val_type alpha_ = (val_type)alpha;
    ptrdiff_t inc = 1;
    ptrdiff_t len = (ptrdiff_t) size;
    val_type *xx = const_cast<val_type*>(x);
    axpy(&len, &alpha_, xx, &inc, y, &inc);
    return y;
}
template<typename val_type, typename T>
dvec_t<val_type>& do_axpy(T alpha, const dvec_t<val_type> &x, dvec_t<val_type> &y) {
    do_axpy(alpha, x.data(), y.data(), x.size());
    return y;
}
template<typename val_type, typename T>
dvec_t<val_type>& do_axpy(T alpha, const svec_t<val_type> &x, dvec_t<val_type> &y) {
    if(alpha == 0) return y;
    for(size_t i = 0; i < x.get_nnz(); i++) {
        y[x.idx[i]] += alpha * x.val[i];
    }
    return y;
}
template<typename XV, typename YV, typename T>
sdvec_t<YV>& do_axpy(T alpha, const svec_t<XV>& x, sdvec_t<YV> &y) {
    if(alpha == 0) return y;
    for(size_t i = 0; i < x.get_nnz(); i++) {
        y.add_nonzero_at(x.idx[i], alpha * x.val[i]);
    }
    return y;
}
template<typename XV, typename YV, typename T>
sdvec_t<YV>& do_axpy(T alpha, const dvec_t<XV>& x, sdvec_t<YV> &y) {
    if(alpha == 0) return y;
    for(size_t i = 0; i < x.size(); i++) {
        y.add_nonzero_at(i, alpha * x[i]);
    }
    return y;
}
template<typename val_type, typename T>
dmat_t<val_type>& do_axpy(T alpha, const dmat_t<val_type> &x, dmat_t<val_type> &y) {
    assert(x.rows == y.rows && x.cols == y.cols);
    if((x.is_rowmajor() && y.is_rowmajor()) || (x.is_colmajor() && y.is_colmajor()))
        do_axpy(alpha, x.data(), y.data(), x.rows*x.cols);
    else {
        if(x.rows > x.cols) {
#pragma omp parallel for schedule(static)
            for(size_t i = 0; i < x.rows; i++)
                for(size_t j = 0; j < x.cols; j++)
                    y.at(i,j) += alpha*x.at(i,j);
        } else {
#pragma omp parallel for schedule(static)
            for(size_t j = 0; j < x.cols; j++)
                for(size_t i = 0; i < x.rows; i++)
                    y.at(i,j) += alpha*x.at(i,j);
        }
    }
    return y;
}

// x *= alpha
template<typename val_type, typename T>
void do_scale(T alpha, val_type *x, size_t size) {
    if(alpha == 0.0) {
        memset(x, 0, sizeof(val_type) * size);
    } else if (alpha == 1.0) {
        return;
    } else {
        val_type alpha_minus_one = (val_type)(alpha - 1);
        do_axpy(alpha_minus_one, x, x, size);
    }
}
template<typename val_type, typename T>
void do_scale(T alpha, dvec_t<val_type> &x) {
    do_scale(alpha, x.data(), x.size());
}
template<typename val_type, typename T>
void do_scale(T alpha, svec_t<val_type> &x) {
    do_scale(alpha, x.val, x.get_nnz());
}
template<typename val_type, typename T>
void do_scale(T alpha, gvec_t<val_type> &x) {
    if(x.is_sparse())
        do_scale(alpha, x.get_sparse());
    else if(x.is_dense())
        do_scale(alpha, x.get_dense());
}
template<typename val_type, typename T>
void do_scale(T alpha, dmat_t<val_type> &x) {
    do_scale(alpha, x.data(), x.rows*x.cols);
}
template<typename val_type, typename T>
void do_scale(T alpha, smat_t<val_type> &x) {
    do_scale(alpha, x.val, x.get_nnz());
    do_scale(alpha, x.val_t, x.get_nnz());
}

// H = a*X*W + b H0 (H0 can put H. However H don't need to be pre-allocated, but H0 do.)
template<typename val_type, typename T2, typename T3>
dmat_t<val_type>& dmat_x_dmat(T2 a, const dmat_t<val_type> &X, const dmat_t<val_type> &W, T3 b, const dmat_t<val_type> &H0, dmat_t<val_type> &H) {
    if(b == 0)
        assert(X.cols == W.rows);
    else
        assert(W.cols == H0.cols && X.cols == W.rows && X.rows == H0.rows);
    H.lazy_resize(X.rows, W.cols).assign(b, H0);

    return dmat_x_dmat(a, X, W, 1, H);
}
template<typename val_type, typename T2, typename T3>
dmat_t<val_type>& smat_x_dmat(T2 a, const smat_t<val_type> &X, const dmat_t<val_type> &W, T3 b, const dmat_t<val_type> &H0, dmat_t<val_type> &H) {
    if(b == 0)
        assert(X.cols == W.rows);
    else
        assert(W.cols == H0.cols && X.cols == W.rows && X.rows == H0.rows);
    H.lazy_resize(X.rows, W.cols).assign(b, H0);

    // H += aXW
    if(W.is_rowmajor()) {
        if(H.is_rowmajor()) {
            smat_x_dmat(a, X, W.data(), W.cols, 1.0, H.data(), H.data());
        } else { // H is col_major
#pragma omp parallel for schedule(dynamic, 64)  shared(X, W, H)
            for(size_t i = 0; i < X.rows; i++) {
                for(size_t idx = X.row_ptr[i]; idx != X.row_ptr[i+1]; idx++){
                    size_t j = X.col_idx[idx];
                    const val_type &Xij = X.val_t[idx];
                    for(size_t t = 0; t < W.cols; t++)
                        H.at(i,t) += a*Xij*W.at(j,t);
                }
            }
        }
    } else { // W.is_colmajor
        if(H.is_colmajor()) {
#pragma omp parallel for schedule(static)
            for(size_t j = 0; j < W.cols; j++)
            {
                dvec_t<val_type> Wj = W.get_col(j);
                dvec_t<val_type> Hj = H.get_col(j);
                X.Xv(Wj, Hj, true);
            }
        } else { // H.is row_major
#pragma omp parallel for schedule(dynamic, 64)  shared(X, W, H)
            for(size_t i = 0; i < X.rows; i++) {
                for(size_t idx = X.row_ptr[i]; idx != X.row_ptr[i+1]; idx++){
                    size_t j = X.col_idx[idx];
                    const val_type &Xij = X.val_t[idx];
                    for(size_t t = 0; t < W.cols; t++)
                        H.at(i,t) += a*Xij*W.at(j,t);
                }
            }
        }
    }
    return H;
}
template<typename val_type, typename T2, typename T3>
dmat_t<val_type>& gmat_x_dmat(T2 a, const gmat_t<val_type> &X, const dmat_t<val_type> &W, T3 b, const dmat_t<val_type> &H0, dmat_t<val_type> &H) {
    if(b == 0)
        assert(X.cols == W.rows);
    else
        assert(W.cols == H0.cols && X.cols == W.rows && X.rows == H0.rows);

    if(X.is_sparse())
        smat_x_dmat(a, X.get_sparse(), W, b, H0, H);
    else if(X.is_dense())
        dmat_x_dmat(a, X.get_dense(), W, b, H0, H);
    else if(X.is_identity()) {
        H.lazy_resize(X.rows, W.cols).assign(b, H0);
        do_axpy(a, W, H);
    }
    return H;
}

// H = a*X*W + H0 (H0 can put H. However H don't need to be pre-allocated, but H0 do)
template<typename val_type, typename T2>
dmat_t<val_type>& dmat_x_dmat(T2 a, const dmat_t<val_type> &X, const dmat_t<val_type> &W, const dmat_t<val_type> &H0, dmat_t<val_type> &H) {
    return dmat_x_dmat(a, X, W, 1.0, H0, H);
}
template<typename val_type, typename T2>
dmat_t<val_type>& smat_x_dmat(T2 a, const smat_t<val_type> &X, const dmat_t<val_type> &W, const dmat_t<val_type> &H0, dmat_t<val_type> &H) {
    return smat_x_dmat(a, X, W, 1.0, H0, H);
}
template<typename val_type, typename T2>
dmat_t<val_type>& gmat_x_dmat(T2 a, const gmat_t<val_type> &X, const dmat_t<val_type> &W, const dmat_t<val_type> &H0, dmat_t<val_type> &H) {
    return gmat_x_dmat(a, X, W, 1.0, H0, H);
}

// H = X*W (H don't need to be pre-allocated)
template<typename val_type>
dmat_t<val_type>& dmat_x_dmat(const dmat_t<val_type> &X, const dmat_t<val_type> &W, dmat_t<val_type> &H) {
    return dmat_x_dmat(1.0, X, W, 0.0, H, H);
}
template<typename val_type>
dmat_t<val_type> operator*(const dmat_t<val_type> &X, const dmat_t<val_type> &W) {
    dmat_t<val_type> H(X.rows, W.cols);
    dmat_x_dmat(X, W, H);
    return H;
}
template<typename VX, typename VW, typename VH>
smat_t<VH>& smat_x_smat_single_thread(const smat_t<VX> &X, const smat_t<VW> &W, smat_t<VH> &H) {
    std::vector<unsigned> row_idx;
    std::vector<size_t> col_ptr;
    std::vector<VH> val;
    size_t rows = X.rows, cols = W.cols;
    sdvec_t<VH> temp(rows);
    col_ptr.push_back(0);
    size_t total_nnz = 0;
    for(size_t c = 0; c < cols; c++) {
        const svec_t<VW>& Wc = W.get_col(c);
        temp.clear();
        for(size_t s = 0; s < Wc.nnz; s++) {
            // temp += Wc[i] * Xi
            do_axpy(Wc.val[s], X.get_col(Wc.idx[s]), temp);
        }
        temp.update_nz_idx();
        total_nnz += temp.nnz;
        col_ptr.push_back(total_nnz);
        for(size_t s = 0; s < temp.nnz; s++) {
            row_idx.push_back(temp.nz_idx[s]);
            val.push_back(temp[temp.nz_idx[s]]);
        }
    }
    H.allocate_space(rows, cols, total_nnz);
    memcpy(H.val, val.data(), sizeof(VH) * total_nnz);
    memcpy(H.row_idx, row_idx.data(), sizeof(unsigned) * total_nnz);
    memcpy(H.col_ptr, col_ptr.data(), sizeof(size_t) * (cols + 1));
    H.csc_to_csr();
    return H;
}
template<typename VX, typename VW, typename VH>
smat_t<VH>& smat_x_smat(const smat_t<VX> &X, const smat_t<VW> &W, smat_t<VH> &H, int threads=-1) {
    struct worker_t {
        worker_t() {}
        sdvec_t<VH> temp;
        std::vector<unsigned> row_idx;
        std::vector<VH> val;

        size_t nnz() const { return row_idx.size(); }

        void set_rows(size_t rows) { temp.resize(rows); }

        void reserve(size_t capacity) {
            row_idx.reserve(capacity);
            val.reserve(capacity);
        }

        void push_back(unsigned idx, VH value) {
            row_idx.push_back(static_cast<unsigned>(idx));
            val.push_back(static_cast<VH>(value));
        }
    };

    size_t rows = X.rows, cols = W.cols;
    if(threads == 1) { return smat_x_smat_single_thread(X, W, H); }

    if(rows > cols) {
        // maximize the parallelism
        smat_t<VX> Xt = X.transpose();
        smat_t<VW> Wt = W.transpose();
        smat_x_smat(Wt, Xt, H, threads);
        H.to_transpose();
        return H;
    }

    if(threads < 1) { threads = omp_get_num_procs(); }
    threads = std::min(threads, omp_get_num_procs());

    std::vector<worker_t> worker_set(threads);
    std::vector<size_t> col_ptr(cols + 1);
    size_t workload = (cols / threads) + (cols % threads != 0);
#pragma omp parallel for schedule(static,1)
    for(int tid = 0; tid < threads; tid++) {
        worker_t& worker = worker_set[tid];
        worker.set_rows(rows);
        worker.reserve(X.nnz + W.nnz);
        size_t c_start = tid * workload;
        size_t c_end = std::min((tid + 1) * workload, cols);
        sdvec_t<VH>& temp = worker.temp;
        for(size_t c = c_start; c < c_end; c++) {
            const svec_t<VW>& Wc = W.get_col(c);
            temp.clear();
            for(size_t s = 0; s < Wc.nnz; s++) {
                // temp += Wc[i] * Xi
                do_axpy(Wc.val[s], X.get_col(Wc.idx[s]), temp);
            }
            temp.update_nz_idx();
            col_ptr[c + 1] = temp.nnz;
            for(size_t s = 0; s < temp.nnz; s++) {
                size_t r = temp.nz_idx[s];
                worker.push_back(r, temp[r]);
            }
        }

    }
    for(size_t c = 1; c <= cols; c++) {
        col_ptr[c] += col_ptr[c - 1];
    }
    size_t total_nnz = col_ptr[cols];

    H.allocate_space(rows, cols, total_nnz);
    memcpy(H.col_ptr, col_ptr.data(), sizeof(size_t) * (cols + 1));
#pragma omp parallel for schedule(static,1)
    for(int tid = 0; tid < threads; tid++) {
        size_t c_start = tid * workload;
        worker_t& worker = worker_set[tid];
        memcpy(&H.val[col_ptr[c_start]], worker.val.data(), sizeof(VH) * worker.nnz());
        memcpy(&H.row_idx[col_ptr[c_start]], worker.row_idx.data(), sizeof(unsigned) * worker.nnz());
    }
    H.csc_to_csr();
    return H;
}
template<typename VX, typename VW>
smat_t<VX> operator*(const smat_t<VX> &X, const smat_t<VW>& W) {
    smat_t<VX> H;
    smat_x_smat(X, W, H);
    return H;
}
template<typename val_type>
dmat_t<val_type>& smat_x_dmat(const smat_t<val_type> &X, const dmat_t<val_type> &W, dmat_t<val_type> &H) {
    return smat_x_dmat(1.0, X, W, 0.0, H, H);
}
template<typename val_type>
dmat_t<val_type> operator*(const smat_t<val_type> &X, const dmat_t<val_type> &W) {
    dmat_t<val_type> H(X.rows, W.cols);
    smat_x_dmat(X, W, H);
    return H;
}
template<typename val_type>
dmat_t<val_type> operator*(const dmat_t<val_type> &X, const smat_t<val_type> &W) {
    dmat_t<val_type> H(X.rows, W.cols);
    smat_x_dmat(X.transpose(), W.transpose(), H.transpose());
    return H;
}
template<typename val_type>
dmat_t<val_type>& gmat_x_dmat(const gmat_t<val_type> &X, const dmat_t<val_type> &W, dmat_t<val_type> &H) {
    return gmat_x_dmat(1.0, X, W, 0.0, H, H);
}
template<typename val_type>
dmat_t<val_type> operator*(const gmat_t<val_type> &X, const dmat_t<val_type> &W) {
    dmat_t<val_type> H(X.rows, W.cols);
    gmat_x_dmat(X, W, H);
    return H;
}
template<typename val_type, typename I, typename V>
void compute_sparse_entries_from_gmat_x_gmat(
        const gmat_t<val_type> &gX, const gmat_t<val_type> &gM,
        size_t len, const I *X_row_idx, const I *M_col_idx, V *val) {
    if(gX.is_sparse() && gM.is_sparse()) {
        const smat_t<val_type>& X = gX.get_sparse();
        const smat_t<val_type>& M = gM.get_sparse();
#pragma omp parallel for schedule(dynamic,64)
        for(size_t idx = 0; idx < len; idx++) {
            const svec_t<val_type>& xi = X.get_row(X_row_idx[idx]);
            const svec_t<val_type>& mj = M.get_col(M_col_idx[idx]);
            val[idx] = static_cast<V>(do_dot_product(xi, mj));
        }
    } else if(gX.is_sparse() && gM.is_dense()) {
        const smat_t<val_type>& X = gX.get_sparse();
        const dmat_t<val_type>& M = gM.get_dense();
#pragma omp parallel for schedule(dynamic,64)
        for(size_t idx = 0; idx < len; idx++) {
            const svec_t<val_type>& xi = X.get_row(X_row_idx[idx]);
            I j = M_col_idx[idx];
            double tmp = 0;
            for(size_t t = 0; t < xi.nnz; t++) {
                tmp += xi.val[t] * M.at(xi.idx[t], j);
            }
            val[idx] = tmp;
        }
    } else if(gX.is_dense() && gM.is_sparse()) {
        const dmat_t<val_type>& X = gX.get_dense();
        const smat_t<val_type>& M = gM.get_sparse();
#pragma omp parallel for schedule(dynamic,64)
        for(size_t idx = 0; idx < len; idx++) {
            const svec_t<val_type>& mj = M.get_col(M_col_idx[idx]);
            I i = X_row_idx[idx];
            double tmp = 0;
            for(size_t t = 0; t < mj.nnz; t++) {
                tmp += X.at(i, mj.idx[t]) * mj.val[t];
            }
            val[idx] = tmp;
        }
    } else if(gX.is_dense() && gM.is_dense()) {
        const dmat_t<val_type>& X = gX.get_dense();
        const dmat_t<val_type>& M = gM.get_dense();
#pragma omp parallel for schedule(static,64)
        for(size_t idx = 0; idx < len; idx++) {
            I i = X_row_idx[idx];
            I j = M_col_idx[idx];
            double tmp = 0;
            for(size_t t = 0; t < X.cols; t++) {
                tmp += X.at(i, t) * M.at(t, j);
            }
            val[idx] = tmp;
        }
    }

}

// tr(W^T X H) (W, H: dense matrix; X: sparse matrix)
template<typename val_type>
val_type trace_dmat_T_smat_dmat(const dmat_t<val_type> &W, const smat_t<val_type> &X, const dmat_t<val_type> &H) {
    assert(W.cols == H.cols && W.rows == X.rows && H.rows == X.cols);
    if(W.is_colmajor() && H.is_colmajor()) {
        double ret = 0;
#pragma omp parallel for schedule(static) reduction(+:ret)
        for(size_t t = 0; t < W.cols; t++) {
            const dvec_t<val_type> u = W.get_col(t);
            const dvec_t<val_type> v = H.get_col(t);
            double local_sum = 0;
            for(size_t i = 0; i < X.rows; i++) {
                for(size_t idx = X.row_ptr[i]; idx != X.row_ptr[i+1]; idx++)
                    local_sum += X.val_t[idx]*u[i]*v[X.col_idx[idx]];
            }
            ret += local_sum;
        }
        return ret;
    } else {
        double ret= 0;
#pragma omp parallel for schedule(dynamic,64) reduction(+:ret)
        for(size_t i = 0; i < X.rows; i++) {
            double  local_sum = 0;
            for(size_t idx = X.row_ptr[i]; idx != X.row_ptr[i+1]; idx++) {
                size_t j = X.col_idx[idx];
                double sum = 0;
                for(size_t t = 0; t < W.cols; t++)
                    sum += W.at(i,t)*H.at(j,t);
                local_sum += sum * X.val_t[idx];
            }
            ret += local_sum;
        }
        return ret;
    }
}

// tr(W^T diag(D) H) (W, H: dense matrix; D: dense vector)
template<typename val_type>
val_type trace_dmat_T_diag_dmat(const dmat_t<val_type> &W, const dvec_t<val_type> &D, const dmat_t<val_type> &H) {
    assert(W.rows == H.rows && W.rows == D.len && W.cols == H.cols);
    assert(W.is_rowmajor() && H.is_rowmajor());
    return trace_dmat_T_diag_dmat(W.data(),D.data(),H.data(),W.rows,W.cols);
}

// -------------- Implementation of Linear Algebra Solvers --------------

// Solve Ax = b, A is symmetric positive definite, b is overwritten with the result x
// A will be modifed by internal Lapack. Make copy when necessary
template<typename val_type>
bool ls_solve_chol(val_type *A, val_type *b, size_t n) {
  ptrdiff_t nn=n, lda=n, ldb=n, nrhs=1, info=0;
  char uplo = 'U';
  posv(&uplo, &nn, &nrhs, A, &lda, b, &ldb, &info);
  return (info == 0);
}

// Solve AX = B, A is symmetric positive definite, B is overwritten with the result X
// A is a m-by-m matrix, while B is a m-by-n matrix stored in col_major
// A will be modified by internal Lapack. Make copy when necessary
template<typename val_type>
bool ls_solve_chol_matrix_colmajor(val_type *A, val_type *B, size_t m, size_t n = size_t(0)) {
  ptrdiff_t mm=m, lda=m, ldb=m, nrhs=n, info=0;
  char uplo = 'U';
  posv(&uplo, &mm, &nrhs, A, &lda, B, &ldb, &info);
  return (info == 0);
}

// Solve AX = B, A is symmetric positive definite, return X
template<typename val_type>
dmat_t<val_type> ls_solve_chol(const dmat_t<val_type>& A, const dmat_t<val_type>& B, bool A_as_workspace) {
    dmat_t<val_type> X(B);
    X.grow_body().to_colmajor();

    dmat_t<val_type> AA(A);
    if(A_as_workspace == false)
        AA.grow_body();

    if(ls_solve_chol_matrix_colmajor(AA.data(), X.data(), AA.rows, X.cols) == false)
        fprintf(stderr, "error when applying ls_solve_cho_matrix_colmajor");
    return X;
}

// Solve Ax = b, A is symmetric positive definite, return x
template<typename val_type>
dvec_t<val_type> ls_solve_chol(const dmat_t<val_type>& A, const dvec_t<val_type>& b, bool A_as_workspace) {
    dvec_t<val_type> x(b);
    x.grow_body();

    dmat_t<val_type> AA(A);
    if(A_as_workspace == false)
        AA.grow_body();

    if(ls_solve_chol(AA.data(), x.data(), AA.rows) == false)
        fprintf(stderr, "error when applying ls_solve_chol");
    return x;
}

// SVD: A = USV'
// U, S, V don't necessarily need to be pre-allocated
template<typename val_type>
class svd_solver_t {
    private:
        char jobz;
        ptrdiff_t mm, nn, min_mn, max_mn, lda, ldu, ldvt, lwork1, lwork2, lwork, info;
        std::vector<val_type> u_buf, v_buf, s_buf, work;
        std::vector<ptrdiff_t> iwork;
        size_t k;

        void prepare_parameter(const dmat_t<val_type>& A, dmat_t<val_type>& U, dvec_t<val_type>& S, dmat_t<val_type>& V, bool reduced) {
            k = std::min(A.rows, A.cols);
            mm = (ptrdiff_t)A.rows;
            nn = (ptrdiff_t)A.cols;
            min_mn = std::min(mm,nn);
            max_mn = std::max(mm,nn);
            lda = mm;
            ldu = mm;
            ldvt = reduced? min_mn : nn;
            lwork1 = 3*min_mn*min_mn + std::max(max_mn, 4*min_mn*min_mn + 4*min_mn);
            lwork2 = 3*min_mn + std::max(max_mn, 4*min_mn*min_mn + 3*min_mn + max_mn);
            lwork = 2 * std::max(lwork1, lwork2);  // due to differences between lapack 3.1 and 3.4
            info = 0;
            work.resize(lwork);
            iwork.resize((size_t)(8*min_mn));
            if(!S.is_view() || S.size() != k)
                S.resize(k);
            if(reduced) {
                jobz = 'S';
                U.lazy_resize(A.rows, k, COLMAJOR);
                V.lazy_resize(A.cols, k, ROWMAJOR);
            } else {
                jobz = 'A';
                U.lazy_resize(A.rows, A.rows, COLMAJOR);
                V.lazy_resize(A.cols, A.cols, ROWMAJOR);
            }
        }

    public:
        svd_solver_t() {}
        bool solve(const dmat_t<val_type>& A, dmat_t<val_type>& U, dvec_t<val_type>& S, dmat_t<val_type>& V, bool reduced=true, bool A_as_workspace=false) {
            if(A.is_rowmajor())
                return solve(A.transpose(), V, S, U, reduced, A_as_workspace);
            else {
                dmat_t<val_type> AA(A.get_view());
                if(A_as_workspace == false)
                    AA.grow_body();
                prepare_parameter(AA, U, S, V, reduced);
#if defined(CPP11)
                gesdd(&jobz, &mm, &nn, AA.data(), &lda, S.data(), U.data(), &ldu, V.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
#else
                gesdd(&jobz, &mm, &nn, AA.data(), &lda, S.data(), U.data(), &ldu, V.data(), &ldvt, &work[0], &lwork, &iwork[0], &info);
#endif
                return (info == 0);
            }
        }
};
template<typename val_type>
void svd(const dmat_t<val_type>& A, dmat_t<val_type>& U, dvec_t<val_type>& S, dmat_t<val_type>& V, bool reduced, bool A_as_workspace) {
    svd_solver_t<val_type> solver;
    solver.solve(A, U, S, V, reduced, A_as_workspace);
}

// -------------- Implementation of Miscellaneous Functions --------------

// y = x for pointer to array
template<typename val_type>
void do_copy(const val_type *x, val_type *y, size_t size) {
    if(x == y) return;
    ptrdiff_t inc = 1;
    ptrdiff_t len = (ptrdiff_t) size;
    val_type *xx = const_cast<val_type*>(x);
    copy(&len, xx, &inc, y, &inc);
}

// H = a*X*W + b H0
// X is an m*n
// W is an n*k, row-majored array
// H is an m*k, row-majored array
template<typename val_type, typename T2, typename T3>
void smat_x_dmat(T2 a, const smat_t<val_type> &X, const val_type *W, const size_t k, T3 b, const val_type *H0, val_type *H) {
    size_t m = X.rows;
    val_type aa = (val_type) a;
    val_type bb = (val_type) b;
    if(a == T2(0)) {
        if(bb == (val_type)0.0){
            memset(H, 0, sizeof(val_type)*m*k);
            return ;
        } else {
            if(H!=H0) {
                do_copy(H0, H, m*k);
                //memcpy(H, H0, sizeof(val_type)*m*k);
            }
            do_scale(bb, H, m*k);
        }
        return;
    }
#pragma omp parallel for schedule(dynamic,64) shared(X, W, H, H0, aa,bb)
    for(size_t i = 0; i < m; i++) {
        val_type *Hi = &H[k*i];
        if(bb == (val_type)0.0)
            memset(Hi, 0, sizeof(val_type)*k);
        else {
            if(Hi!=&H0[k*i])
                do_copy(&H0[k*i], Hi, k);
            do_scale(bb, Hi, k);
        }
        for(size_t idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
            const val_type Xij = X.val_t[idx];
            const val_type *Wj = &W[X.col_idx[idx]*k];
            for(size_t t = 0; t < k; t++)
                Hi[t] += aa*Xij*Wj[t];
        }
    }

}
template<typename val_type, typename T2>
void smat_x_dmat(T2 a, const smat_t<val_type> &X, const val_type* W, const size_t k, const val_type *H0, val_type *H) {
    smat_x_dmat(a, X, W, k, 1.0, H0, H);
}

// C = alpha*A*B + beta*C
// C : m * n, k is the dimension of the middle
// (1) A, B, C are stored in column major!
template<typename val_type, typename T1, typename T2>
void dmat_x_dmat_colmajor(T1 alpha, const val_type *A, bool trans_A, const val_type *B, bool trans_B, T2 beta, val_type *C, size_t m, size_t n, size_t k) {
    ptrdiff_t mm = (ptrdiff_t)m, nn = (ptrdiff_t)n, kk = (ptrdiff_t)k;
    ptrdiff_t lda = trans_A? kk:mm, ldb = trans_B? nn:kk, ldc = mm;
    char transpose = 'T', notranspose = 'N';
    char *transa = trans_A? &transpose: &notranspose;
    char *transb = trans_B? &transpose: &notranspose;
    val_type alpha_ = (val_type) alpha;
    val_type beta_ = (val_type) beta;
    val_type *AA = const_cast<val_type*>(A);
    val_type *BB = const_cast<val_type*>(B);
    gemm(transa, transb, &mm, &nn, &kk, &alpha_, AA, &lda, BB, &ldb, &beta_, C, &ldc);
}
// (2) A, B, C are stored in row major!
template<typename val_type, typename T1, typename T2>
void dmat_x_dmat(T1 alpha, const val_type *A, bool trans_A, const val_type *B, bool trans_B, T2 beta, val_type *C, size_t m, size_t n, size_t k) {
    dmat_x_dmat_colmajor(alpha, B, trans_B, A, trans_A, beta, C, n, m, k);
}

// C = alpha*A*B + beta*C
template<typename val_type, typename T1, typename T2>
dmat_t<val_type>& dmat_x_dmat(T1 alpha, const dmat_t<val_type>& A, const dmat_t<val_type>& B, T2 beta, dmat_t<val_type>& C) {
    assert(A.cols == B.rows);
    C.lazy_resize(A.rows, B.cols);
    if (C.is_rowmajor()) {
        bool trans_A = A.is_rowmajor()? false : true;
        bool trans_B = B.is_rowmajor()? false : true;
        dmat_x_dmat(alpha, A.data(), trans_A, B.data(), trans_B, beta, C.data(), C.rows, C.cols, A.cols);
    } else {
        bool trans_A = A.is_colmajor()? false : true;
        bool trans_B = B.is_colmajor()? false : true;
        dmat_x_dmat_colmajor(alpha, A.data(), trans_A, B.data(), trans_B, beta, C.data(), C.rows, C.cols, A.cols);
    }
    return C;
}

// C = A'*B
// C : m*n, k is the dimension of the middle
// A, B, C are stored in row major!
template<typename val_type>
void dmat_trans_x_dmat(const val_type *A, const val_type *B, val_type *C, size_t m, size_t n, size_t k) {
    bool transpose = true;
    dmat_x_dmat(val_type(1.0), A, transpose, B, !transpose, val_type(0.0), C, m, n, k);
}

// C=A*B
// A, B, C are stored in row major!
template<typename val_type>
void dmat_x_dmat(const val_type *A, const val_type *B, val_type *C, size_t m, size_t n, size_t k) {
    bool trans = true;
    dmat_x_dmat(val_type(1.0), A, !trans, B, !trans, val_type(0.0), C, m, n, k);
}

// Input: an n*k row-major matrix H
// Output: an k*k matrix H^TH
template<typename val_type>
void doHTH(const val_type *H, val_type *HTH, size_t n, size_t k) {
    bool transpose = true;
    dmat_x_dmat_colmajor(val_type(1.0), H, !transpose, H, transpose, val_type(0.0), HTH, k, k, n);
}

/*
    trace(W^T X H)
    X is an m*n, sparse matrix
    W is an m*k, row-majored array
    H is an n*k, row-major
*/
template<typename val_type>
val_type trace_dmat_T_smat_dmat(const val_type *W, const smat_t<val_type> &X, const val_type *H, const size_t k) {
    size_t m = X.rows;
    double ret = 0;
#pragma omp parallel for schedule(dynamic,50) shared(X,H,W) reduction(+:ret)
    for(size_t i = 0; i < m; i++) {
        const val_type *Wi = &W[k*i];
        for(long idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
            const val_type *Hj = &H[X.col_idx[idx]*k];
            double tmp=0;
            for(size_t t = 0; t < k; t++)
                tmp += Wi[t]*Hj[t];
            ret += X.val_t[idx]*tmp;
        }
    }
    return (val_type)ret;
}

/*
    trace(W^T diag(D) H)
    D is an m*1 vector
    W is an m*k, row-majored array
    H is an m*k, row-major array
 */
template<typename val_type>
val_type trace_dmat_T_diag_dmat(const val_type *W, const val_type *D, const val_type *H, const size_t m, const size_t k) {
    val_type *w = const_cast<val_type*>(W);
    val_type *h = const_cast<val_type*>(H);
    val_type *d = const_cast<val_type*>(D);
    double ret = 0.0;
#pragma omp parallel for schedule(static) shared(w,h,d) reduction(+:ret)
    for(size_t i = 0; i < m; i++) {
        val_type *wi = &w[i*k], *hi = &h[i*k];
        ret += do_dot_product(wi, wi, k) * d[i];
    }
    return (val_type)ret;
}
template<typename val_type>
val_type trace_dmat_T_diag_dmat(const dmat_t<val_type> &W, const dmat_t<val_type> &D, const dmat_t<val_type> &H) {
    return trace_dmat_T_diag_dmat(W, dvec_t<val_type>(D.get_view()), H);
}

//------------------ Implementation of zip_it -----------------------

// helpler functions and classes for zip_it
template<class T1, class T2>
struct zip_body {
    T1 x; T2 y;
    zip_body(const zip_ref<T1,T2>& other): x(*other.x), y(*other.y){}
    bool operator<(const zip_body &other) const {return x < other.x;}
    bool operator>(zip_body &other) const {return x > other.x;}
    bool operator==(zip_body &other) const {return x == other.x;}
    bool operator!=(zip_body &other) const {return x != other.x;}
};

template<class T1, class T2>
struct zip_ref {
    T1 *x; T2 *y;
    zip_ref(T1 &x, T2 &y): x(&x), y(&y){}
    zip_ref(zip_body<T1,T2>& other): x(&other.x), y(&other.y){}
    bool operator<(zip_ref other) const {return *x < *other.x;}
    bool operator>(zip_ref other) const {return *x > *other.x;}
    bool operator==(zip_ref other) const {return *x == *other.x;}
    bool operator!=(zip_ref other) const {return *x != *other.x;}
    zip_ref& operator=(zip_ref& other) {
        *x = *other.x; *y = *other.y;
        return *(this);
    }
    zip_ref& operator=(zip_body<T1,T2> other) {
        *x = other.x; *y = other.y;
        return *(this);
    }
};

template<class T1, class T2>
void swap(zip_ref<T1,T2> a, zip_ref<T1,T2> b) {
    std::swap(*(a.x),*(b.x));
    std::swap(*(a.y),*(b.y));
}

template<class IterT1, class IterT2>
struct zip_it {
    typedef std::random_access_iterator_tag iterator_category;
    typedef typename std::iterator_traits<IterT1>::value_type T1;
    typedef typename std::iterator_traits<IterT2>::value_type T2;
    typedef zip_body<T1,T2> value_type;
    typedef zip_ref<T1,T2> reference;
    typedef zip_body<T1,T2>* pointer;
    typedef ptrdiff_t difference_type;
    IterT1 x;
    IterT2 y;
    zip_it(IterT1 x, IterT2 y): x(x), y(y){}
    reference operator*() {return reference(*x, *y);}
    reference operator[](const difference_type n) const {return reference(x[n],y[n]);}
    zip_it& operator++() {++x; ++y; return *this;} // prefix ++
    zip_it& operator--() {--x; --y; return *this;} // prefix --
    zip_it operator++(int) {return zip_it(x++,y++);} // sufix ++
    zip_it operator--(int) {return zip_it(x--,y--);} // sufix --
    zip_it operator+(const difference_type n) {return zip_it(x+n,y+n);}
    zip_it operator-(const difference_type n) {return zip_it(x-n,y-n);}
    zip_it& operator+=(const difference_type n) {x+=n; y+=n; return *this;}
    zip_it& operator-=(const difference_type n) {x-=n; y-=n; return *this;}
    bool operator<(const zip_it& other) {return x<other.x;}
    bool operator>(const zip_it& other) {return x>other.x;}
    bool operator==(const zip_it& other) {return x==other.x;}
    bool operator!=(const zip_it& other) {return x!=other.x;}
    difference_type operator-(const zip_it& other) {return x-other.x;}
};

template<class IterT1, class IterT2>
zip_it<IterT1, IterT2> zip_iter(IterT1 x, IterT2 y) {
    return zip_it<IterT1,IterT2>(x,y);
}

// ---------------- Implementation of string split utility --------------

// split utility
template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss; ss.str(s); std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

/*
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

std::vector<std::string>& split(const std::string &s, char delim, std::vector<std::string>& elems) {
    elems.clear();
    split(s, delim, std::back_inserter(elems));
    return elems;
}
*/


#undef coo_t
#undef gmat_t
#undef eye_t
#undef smat_t
#undef dmat_t
#undef gvec_t
#undef sdvec_t
#undef svec_t
#undef dvec_t

// C Interface
extern "C" {
    enum {
        DENSE_ROWMAJOR = 1,
        DENSE_COLMAJOR = 2,
        SPARSE = 3,
        EYE = 4
    };

    typedef struct {
        uint64_t rows, cols, nnz;
        size_t* row_ptr;
        size_t* col_ptr;
        uint32_t* row_idx;
        uint32_t* col_idx;
        void* val;
        void* val_t;
        int32_t type;
    } PyMatrix;
} // end of extern "C"

template<typename val_type>
class general_matrix_wrapper {
    public:
        typedef sparse_vector<val_type> svec_t;
        typedef dense_vector<val_type> dvec_t;
        typedef sparse_dense_vector<val_type> sdvec_t;
        typedef general_vector<val_type> gvec_t;
        typedef sparse_matrix<val_type> smat_t;
        typedef dense_matrix<val_type> dmat_t;
        typedef identity_matrix<val_type> eye_t;
        typedef general_matrix<val_type> gmat_t;
        typedef coo_matrix<val_type> coo_t;

        general_matrix_wrapper() {}
        general_matrix_wrapper(const PyMatrix* py_mat_ptr) {
            if(py_mat_ptr->type == DENSE_ROWMAJOR) {
                dense = dmat_t(py_mat_ptr->rows, py_mat_ptr->cols, ROWMAJOR, static_cast<val_type*>(py_mat_ptr->val));
                gmat_ptr = &dense;
            } else if(py_mat_ptr->type == DENSE_COLMAJOR) {
                dense = dmat_t(py_mat_ptr->rows, py_mat_ptr->cols, COLMAJOR, static_cast<val_type*>(py_mat_ptr->val));
                gmat_ptr = &dense;
            } else if(py_mat_ptr->type == SPARSE) {
                sparse = smat_t(
                            py_mat_ptr->rows, py_mat_ptr->cols, py_mat_ptr->nnz,
                            static_cast<val_type*>(py_mat_ptr->val),
                            static_cast<val_type*>(py_mat_ptr->val_t),
                            py_mat_ptr->col_ptr, py_mat_ptr->row_ptr,
                            py_mat_ptr->row_idx, py_mat_ptr->col_idx);
                gmat_ptr = &sparse;
            }
        }
        size_t rows() const { return gmat_ptr->rows; }
        size_t cols() const { return gmat_ptr->cols; }
        gmat_t& get_gmat() { return *gmat_ptr; }
        const gmat_t& get_gmat() const { return *gmat_ptr; }

        bool is_sparse() const { return gmat_ptr->is_sparse(); }
        bool is_dense() const { return gmat_ptr->is_dense(); }
        bool is_identity() const { return gmat_ptr->is_identity(); }

        smat_t& get_sparse() { return gmat_ptr->get_sparse(); }
        const smat_t& get_sparse() const { return gmat_ptr->get_sparse(); }

        dmat_t& get_dense() { return gmat_ptr->get_dense(); }
        const dmat_t& get_dense() const { return gmat_ptr->get_dense(); }

        general_matrix_wrapper<val_type> transpose() const {
            general_matrix_wrapper gmw;
            gmw.dense = this->dense.transpose();
            if(is_sparse()) {
                gmw.sparse = this->sparse.transpose();
                gmw.gmat_ptr = &gmw.sparse;
            } else if(is_dense()) {
                gmw.dense = this->dense.transpose();
                gmw.gmat_ptr = &gmw.dense;
            } else if(is_identity()) {
                gmw.eye = this->eye;
                gmw.gmat_ptr = &gmw.eye;
            }
            return gmw;
        }

    private:
        smat_t sparse;
        dmat_t dense;
        eye_t eye;
        gmat_t* gmat_ptr;
};

#endif // RF_MATRIX_H
