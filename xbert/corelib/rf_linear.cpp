#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <set>
#include <vector>
#include <time.h>
#include <cstddef>

#include "rf_matrix.h"

typedef size_t IndexType;
#ifndef ValueType
#define ValueType float
#endif

typedef general_matrix_wrapper<ValueType> gmat_wrapper_t;
typedef gmat_wrapper_t::gvec_t gvec_t;
typedef gmat_wrapper_t::dvec_t dvec_t;
typedef gmat_wrapper_t::svec_t svec_t;
typedef gmat_wrapper_t::sdvec_t sdvec_t;
typedef gmat_wrapper_t::gmat_t gmat_t;
typedef gmat_wrapper_t::dmat_t dmat_t;
typedef gmat_wrapper_t::smat_t smat_t;
typedef gmat_wrapper_t::coo_t coo_t;
typedef std::vector<IndexType> ivec_t;
//typedef dense_vector<uint32_t> ivec_t;

extern "C" {
    typedef void(*py_pred_allocator_t)(uint64_t, uint64_t, uint64_t, void*, void*, void*, void*);
}

struct walltime_clock_t {
    int64_t last_time;
    walltime_clock_t(): last_time(0) {}

    int64_t tic() { return last_time = now(); }
    double toc() { return static_cast<double>(now() - last_time) / 1e6; }
    int64_t now() {
        struct timespec tw;
        clock_gettime(CLOCK_MONOTONIC, &tw);
        return tw.tv_sec * (1000000000L) + tw.tv_nsec;
    }
};

//extern int mkl_set_num_threads_local(int);

enum {
    L2R_LR=0,
    L2R_L2LOSS_SVC_DUAL=1,
    L2R_L2LOSS_SVC=2,
    L2R_L1LOSS_SVC_DUAL=3,
    MCSVM_CS=4,
    L1R_L2LOSS_SVC=5,
    L1R_LR=6,
    L2R_LR_DUAL=7
}; /* solver_type */

enum {
    KMEANS=0,
    KDTREE=1,
    SKMEANS=5,
    KDTREE_CYCLIC=11,
}; /* partition_algo */


struct Node {
    size_t start;
    size_t end;

    Node(size_t start=0, size_t end=0): start(start), end(end) {}

    void set(size_t start, size_t end) {
        this->start = start;
        this->end = end;
    }

    size_t size() const { return end - start; }
};

/*
 * Each node is a cluster of elements
 * #leaf nodes = 2^{depth}
 * #internal nodes = 2^{depth} (where we have a dummy node with node Id = 0)
 * #nodes = 2^{depth + 1}
 */
struct Tree {
    size_t depth;     // # leaf nodes = 2^depth
    std::vector<Node> nodes;


    // used for balanced 2-means
    ivec_t elements;
    ivec_t previous_elements;
    dvec_t center1; // need to be duplicated to handle parallel clustering
    dvec_t center2;// for spherical kmeans
    dvec_t scores; // need to be duplicated to handle parallel clustering
    std::vector<unsigned> seed_for_nodes; // random seeds used for each node


    Tree(size_t depth=0) { this->reset_depth(depth); }

    void reset_depth(size_t depth) {
        this->depth = depth;
        nodes.resize(1 << (depth + 1));
        seed_for_nodes.resize(nodes.size());
    }

    struct comparator_by_value_t { // {{{
        const ValueType *pred_val;
        bool increasing;
        comparator_by_value_t(const ValueType *val, bool increasing=true):
            pred_val(val), increasing(increasing) {}
        bool operator()(const size_t i, const size_t j) const {
            if(increasing) {
                return (pred_val[i] < pred_val[j]) || (pred_val[i] == pred_val[j] && i < j);
            } else {
                return (pred_val[i] > pred_val[j]) || (pred_val[i] == pred_val[j] && i < j);
            }
        }
    }; // }}}


    Node& root_of(size_t nid) { return nodes[nid]; }
    Node& left_of(size_t nid) { return nodes[nid << 1]; }
    Node& right_of(size_t nid) { return nodes[(nid << 1) + 1]; }

    void partition_elements(Node& root, Node& left, Node& right) {
        size_t middle = (root.start + root.end) >> 1;
        left.set(root.start, middle);
        right.set(middle, root.end);
    }

    // return true if this sorting changes the assignment, false otherwise.
    bool sort_elements_by_scores_on_node(const Node& root, bool increasing=true) {
        auto prev_start_it = previous_elements.begin() + root.start;
        auto start_it = elements.begin() + root.start;
        auto middle_it = elements.begin() + ((root.start + root.end) >> 1);
        auto end_it = elements.begin() + root.end;
        std::copy(start_it, middle_it, prev_start_it);
        std::sort(start_it, end_it, comparator_by_value_t(scores.data(), increasing));
        std::sort(start_it, middle_it);
        std::sort(middle_it, end_it);
        return !std::equal(start_it, middle_it, prev_start_it);
    }

    // X = [x_1, ..., x_L]^T
    // c_1 = e_1^T X / |e_1|_0, where \be_1 is the indicator for first half elements
    // c_2 = e_2^T X / |e_2|_0, where \be_2 is the indicator for second half elements
    // e = e_2/|e_2|_0 - e_1/|e_1|_0
    // c = c_2 - c_1 = X^T e
    // score(i) = <c, x_i>
    // works for both cosine similarily if feat_mat is with unit-length rows
    //                euclidean similarity

    template<typename MAT>
    void partition_kmeans(size_t nid, size_t depth, const MAT& feat_mat, size_t max_iter=10) {
        Node& root = root_of(nid);
        Node& left = left_of(nid);
        Node& right = right_of(nid);
        partition_elements(root, left, right);
        rng_t rng(seed_for_nodes[nid]);

        auto& cur_center  = center1;
        // perform the clustering and sorting
        for(size_t iter = 0; iter < max_iter; iter++) {
            // construct center (for right child)
            memset(cur_center.buf, 0, sizeof(ValueType) * cur_center.len);
            if(iter == 0) {
                auto right_idx = rng.randint(0, root.size() - 1);
                auto left_idx = (right_idx + rng.randint(1, root.size() - 1)) % root.size();
                right_idx += root.start;
                left_idx  += root.start;

                const auto& feat_right = feat_mat.get_row(elements[right_idx]);
                const auto& feat_left = feat_mat.get_row(elements[left_idx]);
                do_axpy(1.0, feat_right, cur_center);
                do_axpy(-1.0, feat_left, cur_center);

            } else {
                ValueType alpha = 0;
                alpha = +1.0 / right.size();
                for(size_t i = right.start; i < right.end; i++) {
                    size_t eid = elements[i];
                    const auto& feat = feat_mat.get_row(eid);
                    do_axpy(alpha, feat, cur_center);
                }

                alpha = -1.0 / left.size();
                for(size_t i = left.start; i < left.end; i++) {
                    size_t eid = elements[i];
                    const auto& feat = feat_mat.get_row(eid);
                    do_axpy(alpha, feat, cur_center);
                }
            }
            ivec_t *elements_ptr = &elements;
            dvec_t *scores_ptr = &scores;
            dvec_t *center_ptr = &cur_center;
            const MAT* feat_mat_ptr = &feat_mat;
            // construct scores
#pragma omp parallel for shared(elements_ptr, scores_ptr, center_ptr, feat_mat_ptr)
            for(size_t i = root.start; i < root.end; i++) {
                size_t eid = elements_ptr->at(i);
                const svec_t& feat = feat_mat_ptr->get_row(eid);
                scores_ptr->at(eid) = do_dot_product(*center_ptr, feat);
            }
            bool assignment_changed = sort_elements_by_scores_on_node(root);
            if(!assignment_changed) {
                break;
            }
        }
    }

    template<typename MAT>
    void partition_skmeans(size_t nid, size_t depth, const MAT& feat_mat, size_t max_iter=10) {
        Node& root = root_of(nid);
        Node& left = left_of(nid);
        Node& right = right_of(nid);
        partition_elements(root, left, right);
        rng_t rng(seed_for_nodes[nid]);

        auto& cur_center1 = center1;
        auto& cur_center2 = center2;
        // perform the clustering and sorting
        for(size_t iter = 0; iter < max_iter; iter++) {
            ValueType one = 1.0;
            // construct center (for right child)
            memset(cur_center1.buf, 0, sizeof(ValueType) * cur_center1.len);
            memset(cur_center2.buf, 0, sizeof(ValueType) * cur_center2.len);

            if(iter == 0) {
                auto right_idx = rng.randint(0, root.size() - 1);
                auto left_idx = (right_idx + rng.randint(1, root.size() - 1)) % root.size();
                right_idx += root.start;
                left_idx  += root.start;

                const auto& feat_right = feat_mat.get_row(elements[right_idx]);
                const auto& feat_left = feat_mat.get_row(elements[left_idx]);
                do_axpy(1.0, feat_right, cur_center1);
                do_axpy(1.0, feat_left, cur_center2);
                do_axpy(-1.0, cur_center2, cur_center1);
            } else {
                for(size_t i = right.start; i < right.end; i++) {
                    size_t eid = elements[i];
                    const auto& feat = feat_mat.get_row(eid);
                    do_axpy(one, feat, cur_center1);
                }
                ValueType alpha = do_dot_product(cur_center1, cur_center1);
                if(alpha > 0) {
                    do_scale(1.0 / sqrt(alpha), cur_center1);
                }

                for(size_t i = left.start; i < left.end; i++) {
                    size_t eid = elements[i];
                    const auto& feat = feat_mat.get_row(eid);
                    do_axpy(one, feat, cur_center2);
                }
                alpha = do_dot_product(cur_center2, cur_center2);
                if(alpha > 0) {
                    do_scale(1.0 / sqrt(alpha), cur_center2);
                }

                do_axpy(-1.0, cur_center2, cur_center1);
            }
            ivec_t *elements_ptr = &elements;
            dvec_t *scores_ptr = &scores;
            dvec_t *center_ptr = &cur_center1;
            const MAT* feat_mat_ptr = &feat_mat;
            // construct scores
#pragma omp parallel for shared(elements_ptr, scores_ptr, center_ptr, feat_mat_ptr)
            for(size_t i = root.start; i < root.end; i++) {
                size_t eid = elements_ptr->at(i);
                const svec_t& feat = feat_mat_ptr->get_row(eid);
                scores_ptr->at(eid) = do_dot_product(*center_ptr, feat);
            }
            bool assignment_changed = sort_elements_by_scores_on_node(root);
            if(!assignment_changed) {
                break;
            }
        }
    }

    void partition_kdtree(size_t nid, size_t feat_id, const dmat_t& feat_mat) {
        Node& root = root_of(nid);
        Node& left = left_of(nid);
        Node& right = right_of(nid);
        partition_elements(root, left, right);

        for(unsigned i = root.start; i < root.end; i++) {
            scores[i] = feat_mat.at(i, feat_id);
        }

        sort_elements_by_scores_on_node(root);

        for(unsigned i = root.start; i < root.end; i++) {
            scores[i] = 0;
        }
    }

    void partition_kdtree(size_t nid, size_t feat_id, const smat_t& feat_mat) {
        Node& root = root_of(nid);
        Node& left = left_of(nid);
        Node& right = right_of(nid);
        partition_elements(root, left, right);

        // perform the clustering and sorting
        const auto& feat_j = feat_mat.get_col(feat_id);
        for(unsigned t = 0; t < feat_j.nnz; t++) {
            size_t i = feat_j.idx[t];
            if((root.start <= i) && (i < root.end)) {
                scores[i] = feat_j.val[t];
            }
        }

        sort_elements_by_scores_on_node(root);

        for(unsigned t = 0; t < feat_j.nnz; t++) {
            size_t i = feat_j.idx[t];
            if((root.start <= i) && (i < root.end)) {
                scores[i] = 0;
            }
        }
    }

    template<typename IND=unsigned>
    void run_clustering(const gmat_wrapper_t& feat_mat, int partition_algo, int seed=0, IND *label_codes=NULL, size_t max_iter=10) {
        if(feat_mat.is_sparse()) {
            run_clustering(feat_mat.get_sparse(), partition_algo, seed, label_codes, max_iter);
        } else if(feat_mat.is_dense()) {
            run_clustering(feat_mat.get_dense(), partition_algo, seed, label_codes, max_iter);
        } else {
            fprintf(stderr, "Not supported feat_mat type in run_clustering\n");
        }
    }

    template<typename MAT, typename IND=unsigned>
    void run_clustering(const MAT& feat_mat, int partition_algo, int seed=0, IND *label_codes=NULL, size_t max_iter=10) {
        size_t nr_elements = feat_mat.rows;
        elements.resize(nr_elements);
        previous_elements.resize(nr_elements);
        // random shuffle here
        for(size_t i = 0; i < nr_elements; i++) {
            elements[i] = i;
        }
        rng_t rng = rng_t(seed);
        for(size_t nid = 0; nid < nodes.size(); nid++) {
            seed_for_nodes[nid] = rng.randint<unsigned>();
        }

        center1.resize(feat_mat.cols, 0);
        center2.resize(feat_mat.cols, 0);
        scores.resize(feat_mat.rows, 0);
        nodes[0].set(0, nr_elements);
        nodes[1].set(0, nr_elements);

        // let's do it layer by layer so we can parallelize it
        for(size_t d = 0; d < depth; d++) {
            size_t layer_start = 1U << d;
            size_t layer_end = 1U << (d + 1);
            for(size_t nid = layer_start; nid < layer_end; nid++) {
                //fprintf(stderr, ">>>>> depth %d nid %d\n", d, nid);
                if(partition_algo == KMEANS) {
                    partition_kmeans(nid, d, feat_mat, max_iter);
                } else if(partition_algo == SKMEANS) {
                    partition_skmeans(nid, d, feat_mat, max_iter);
                } else if(partition_algo == KDTREE) {
                    partition_kdtree(nid, d % feat_mat.cols, feat_mat);
                } else if(partition_algo == KDTREE_CYCLIC) {
                    size_t nr_repeats = std::max(1UL, static_cast<size_t>(depth / feat_mat.cols));
                    partition_kdtree(nid, (d / nr_repeats) % feat_mat.cols, feat_mat);
                }
            }
        }
        /* an alternative one
        size_t nr_internal_nodes = nodes.size() >> 1;
        for(size_t nid = 1; nid < nr_internal_nodes; nid++) {
            partition(nid, feat_mat);
        }
        */

        if(label_codes != NULL) {
            size_t leaf_start = 1U << depth;
            size_t leaf_end = 1U << (depth + 1);
            for(size_t nid = leaf_start; nid < leaf_end; nid++) {
                for(size_t idx = nodes[nid].start; idx < nodes[nid].end; idx++) {
                    label_codes[elements[idx]] = nid - leaf_start;
                }
            }
        }
    }


    void output() {
        size_t nr_internal_nodes = nodes.size() >> 1;
        for(size_t nid = nr_internal_nodes; nid < nodes.size(); nid++) {
            const Node& node = nodes[nid];
            printf("node(%ld): ", nid);
            for(size_t idx = node.start; idx < node.end; idx++) {
                printf(" %ld", elements[idx]);
            }
            puts("");
        }
    }
};


struct SVMParameter {
    SVMParameter(
        int solver_type=L2R_L1LOSS_SVC_DUAL,
        double Cp=1.0,
        double Cn=1.0,
        int max_iter=1000,
        double eps=0.1,
        double bias=1.0
    ): solver_type(solver_type), max_iter(max_iter), Cp(Cp), Cn(Cn), eps(eps), bias(bias) {}

    int solver_type;
    size_t max_iter;
    double Cp, Cn, eps, bias;
};

#define INF HUGE_VAL
#undef GETI
#define GETI(i) (static_cast<int>(y[i]) + 1)
struct SVMWorker {
    SVMParameter param;
    ivec_t index; // used to determine the subset of rows of X are used in the training.
    ivec_t w_index; // used to determine the subset of active index.
    dvec_t y;
    dvec_t w;
    ValueType bb; // bias parameter
    dvec_t QD;
    dvec_t alpha;
    dvec_t xj_sq;
    svec_t bias_column_vec;
    double upper_bound[3];
    double diag[3];
    size_t w_size, y_size;

    smat_t subX; // used to capture the sub matrix indexed by row indices in index
    std::vector<unsigned> csc_row_idx;
    std::vector<size_t> csc_col_ptr;
    std::vector<ValueType> csc_val;
    // we might need an rng for each workspace

    SVMWorker(): w_size(0), y_size(0) {}

    void init(size_t w_size, size_t y_size, const SVMParameter *param_ptr=NULL) {
        if(param_ptr != NULL) {
            param = *param_ptr;
        }
        this->w_size = w_size;
        this->y_size = y_size;
        w.resize(w_size, 0);
        y.resize(y_size, 0);

        if(param.solver_type == L2R_L2LOSS_SVC_DUAL) {
            alpha.resize(y_size, 0);
            QD.resize(y_size, 0);
        } else if(param.solver_type == L2R_L1LOSS_SVC_DUAL) {
            alpha.resize(y_size, 0);
            QD.resize(y_size, 0);
        } else if(param.solver_type == L2R_LR_DUAL) {
            alpha.resize(2 * y_size, 0);
            QD.resize(y_size, 0);
        } else if(param.solver_type == L1R_L2LOSS_SVC) {
            xj_sq.resize(w_size + 1, 0);
            alpha.resize(y_size); // used as b in l1r_l2loss_svc
            bias_column_vec.resize(y_size, y_size);
        }
    }

    void lazy_init(size_t w_size, size_t y_size, const SVMParameter *param_ptr=NULL) {
        if((w_size != this->w_size)
                || (y_size != this->y_size)
                || ((param_ptr != NULL) && (param_ptr->solver_type != param.solver_type))) {
            init(w_size, y_size, param_ptr);
        } else {
            param = *param_ptr;
        }
    }

    const smat_t& get_sub_feat_mat(const smat_t& X) {
       if(index.size() == y_size ||
                ((param.solver_type != L1R_L2LOSS_SVC)
                 && (param.solver_type != L1R_LR))) {
            return X;
        }

        csc_col_ptr.clear();
        size_t nnz = 0;
        for(size_t c = 0; c <= X.cols; c++) {
            csc_col_ptr.push_back(0);
        }
        for(size_t s = 0; s < index.size(); s++) {
            size_t r = index[s];
            for(size_t idx = X.row_ptr[r]; idx < X.row_ptr[r + 1]; idx++) {
                size_t c = (size_t) X.col_idx[idx];
                csc_col_ptr[c + 1]++;
                nnz++;
            }
        }
        for(size_t c = 1; c <= X.cols; c++) {
            csc_col_ptr[c] += csc_col_ptr[c - 1];
        }

        csc_row_idx.resize(nnz);
        csc_val.resize(nnz);
        for(size_t s = 0; s < index.size(); s++) {
            size_t r = index[s];
            for(size_t idx = X.row_ptr[r]; idx < X.row_ptr[r + 1]; idx++) {
                size_t c = (size_t) X.col_idx[idx];
                csc_row_idx[csc_col_ptr[c]] = r;
                csc_val[csc_col_ptr[c]++] = X.val_t[idx];
            }
        }
        for(size_t c = X.cols; c > 0; c--) {
            csc_col_ptr[c] = csc_col_ptr[c - 1];
        }
        csc_col_ptr[0] = 0;
        subX.rows = X.rows;
        subX.cols = X.cols;
        subX.nnz = nnz;
        subX.row_idx = csc_row_idx.data();
        subX.val = csc_val.data();
        subX.col_ptr = csc_col_ptr.data();
        subX.col_idx = X.col_idx;
        subX.row_ptr = X.row_ptr;
        subX.val_t = X.val_t;
        return subX;
    }

    dvec_t& solve(const gmat_wrapper_t& feat_mat, int seed=0) {
        // the solution will be available in w and bb
        if(feat_mat.is_sparse()) {
            solve(feat_mat.get_sparse(), seed);
        } else  {
            fprintf(stderr, "Non sparse feat_matrix is not supported yet\n");
            //solve(feat_mat.get_dense(), seed);
        }
        return w;
    }

    template<typename MAT>
    dvec_t& solve(const MAT& X, int seed=0) {
        if(param.solver_type == L2R_L1LOSS_SVC_DUAL) {
            solve_l2r_l1l2_svc(X, seed);
        } else if(param.solver_type == L2R_L2LOSS_SVC_DUAL) {
            solve_l2r_l1l2_svc(X, seed);
        } else if(param.solver_type == L2R_LR_DUAL) {
            solve_l2r_lr(X, seed);
        } else if(param.solver_type == L1R_L2LOSS_SVC) {
            const MAT& XX = get_sub_feat_mat(X);
            solve_l1r_l2svc(XX, seed);
        }
        //compute_obj();
        return w;
    }

    template<typename T1, typename T2>
    void do_axpy_with_bias(ValueType a, const T1& x, const ValueType& x_bias, T2& y, ValueType& y_bias, const SVMParameter& param) {
        do_axpy(a, x, y);
        if(param.bias > 0) {
            y_bias += a * x_bias;
        }
    }

    template<typename T1, typename T2>
    ValueType do_dot_product_with_bias(const T1& x, const ValueType& x_bias, const T2& y, const ValueType& y_bias, const SVMParameter& param) {
        ValueType ret = do_dot_product(x, y);
        if(param.bias > 0) {
            ret += x_bias * y_bias;
        }
        return ret;
    }

    template<typename MAT>
    void solve_l2r_l1l2_svc(const MAT& X, int seed) {

        rng_t rng(seed);

        for(size_t j = 0; j < w_size; j++) {
            w[j] = 0;
        }
        bb = 0;

        if(param.solver_type == L2R_L2LOSS_SVC_DUAL) {
            diag[0] = 0.5 / param.Cn;
            diag[2] = 0.5 / param.Cp;
            upper_bound[0] = INF;
            upper_bound[2] = INF;
        } else if(param.solver_type == L2R_L1LOSS_SVC_DUAL) {
            diag[0] = 0;
            diag[2] = 0;
            upper_bound[0] = param.Cn;
            upper_bound[2] = param.Cp;
        }


        for(auto& i : index) {
            alpha[i] = 0;
            QD[i] = diag[GETI(i)];

            const svec_t& xi = X.get_row(i);
            QD[i] += do_dot_product_with_bias(xi, param.bias, xi, param.bias, param);
            double coef = y[i] * alpha[i];
            do_axpy_with_bias(coef, xi, param.bias, w, bb, param);
        }

        // PG: projected gradient, for shrinking and stopping
        double PGmax_old = INF;
        double PGmin_old = -INF;
        double PGmax_new, PGmin_new;

        size_t active_size = index.size();
        size_t iter = 0;
        while(iter < param.max_iter) {
            PGmax_new = -INF;
            PGmin_new = INF;

            // shuffle
            rng.shuffle(index.begin(), index.begin() + active_size);

            size_t s = 0;
            for(s = 0; s < active_size; s++) {
                size_t i = index[s];
                const signed char yi = y[i];
                const svec_t& xi = X.get_row(i);

               // double G = yi * (do_dot_product(w, xi) + (param.bias > 0 ? bb * param.bias : 0.0))- 1;
                double G = yi * do_dot_product_with_bias(w, bb, xi, param.bias, param) - 1.0;
                double C = upper_bound[GETI(i)];
                G += alpha[i] * diag[GETI(i)];

                double PG = 0;
                if(alpha[i] == 0) {
                    if(G > PGmax_old) {
                        active_size--;
                        std::swap(index[s], index[active_size]);
                        s--;
                        continue;
                    } else if (G < 0) {
                        PG = G;
                    }
                } else if (alpha[i] == C) {
                    if (G < PGmin_old) {
                        active_size--;
                        std::swap(index[s], index[active_size]);
                        s--;
                        continue;
                    } else if (G > 0) {
                        PG = G;
                    }
                } else {
                    PG = G;
                }

                PGmax_new = std::max(PGmax_new, PG);
                PGmin_new = std::min(PGmin_new, PG);

                if(fabs(PG) > 1.0e-12) {
                    double alpha_old = alpha[i];
                    alpha[i] = static_cast<ValueType>(std::min(std::max(alpha[i] - G / QD[i], 0.0), C));
                    double d = (alpha[i] - alpha_old) * yi;
                    do_axpy_with_bias(d, xi, param.bias, w, bb, param);
                }
            }

            iter++;
            if(PGmax_new - PGmin_new <= param.eps) {
                if(active_size == index.size()) {
                    break;
                } else {
                    active_size = index.size();
                    PGmax_old = INF;
                    PGmin_old = -INF;
                    continue;
                }
            }
            PGmax_old = PGmax_new;
            PGmin_old = PGmin_new;
            if (PGmax_old <= 0) {
                PGmax_old = INF;
            }
            if (PGmin_old >= 0) {
                PGmin_old = -INF;
            }
        }
    }

    template<typename MAT>
    void solve_l2r_lr(const MAT& X, int seed) {
        rng_t rng(seed);

        dvec_t& xTx = QD;
        size_t max_inner_iter = 100; // for inner Newton
        double innereps = 1e-2;
        double innereps_min = std::min(1e-8, param.eps);

        upper_bound[0] = param.Cn;
        upper_bound[2] = param.Cp;

        for(size_t j = 0; j < w_size; j++) {
            w[j] = 0;
        }
        bb = 0;

        // Initial alpha can be set here. Note that
        // 0 < alpha[i] < upper_bound[GETI(i)]
        // alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
        for(auto& i : index) {
            alpha[2 * i] = std::min(0.001 * upper_bound[GETI(i)], 1e-8);
            alpha[2 * i + 1] = upper_bound[GETI(i)] - alpha[2 * i];

            const svec_t& xi = X.get_row(i);
            xTx[i] += do_dot_product_with_bias(xi, param.bias, xi, param.bias, param);
            double coef = y[i] * alpha[2 * i];
            do_axpy_with_bias(coef, xi, param.bias, w, bb, param);
        }

        size_t iter = 0;
        while(iter < param.max_iter) {
            // shuffle
            rng.shuffle(index.begin(), index.end());

            size_t newton_iter = 0;
            double Gmax = 0;
            for(auto& i : index) {
                const signed char yi = y[i];
                const svec_t& xi = X.get_row(i);

                double C = upper_bound[GETI(i)];
                double xisq = xTx[i];
                double ywTx = yi * do_dot_product_with_bias(w, bb, xi, param.bias, param);
                double a = xisq, b = ywTx;

                // Decide to minimize g_1(z) or g_2(z)
                int ind1 = 2 * i, ind2 = 2 * i+1, sign = 1;
                if(0.5 * a * (alpha[ind2] - alpha[ind1]) + b < 0) {
                    ind1 = 2 * i + 1;
                    ind2 = 2 * i;
                    sign = -1;
                }

                //  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
                double alpha_old = alpha[ind1];
                double z = alpha_old;
                if(C - z < 0.5 * C) {
                    z = 0.1 * z;
                }
                double gp = a * (z - alpha_old) + sign * b + log(z / (C - z));
                Gmax = std::max(Gmax, fabs(gp));

                // Newton method on the sub-problem
                const double eta = 0.1; // xi in the paper
                size_t inner_iter = 0;
                while(inner_iter <= max_inner_iter) {
                    if(fabs(gp) < innereps) {
                        break;
                    }
                    double gpp = a + C/(C-z)/z;
                    double tmpz = z - gp/gpp;
                    if(tmpz <= 0) {
                        z *= eta;
                    } else { // tmpz in (0, C)
                        z = tmpz;
                    }
                    gp = a * (z - alpha_old) + sign * b + log(z / (C - z));
                    newton_iter++;
                    inner_iter++;
                }
                if(inner_iter > 0) { // update w
                    alpha[ind1] = z;
                    alpha[ind2] = C - z;
                    double coef = sign * (z - alpha_old) * yi;
                    do_axpy_with_bias(coef, xi, param.bias, w, bb, param);
                }
            }

            iter++;
            if(Gmax < param.eps) {
                break;
            }

            if(newton_iter <= index.size() / 10) {
                innereps = std::max(innereps_min, 0.1 * innereps);
            }
        }
    }

	template<typename MAT>
	void solve_l1r_l2svc(const MAT& X, int seed) {

        rng_t rng(seed);

        size_t max_num_linesearch = 20;
        double sigma = 0.01; // coef for line search condition
        dvec_t& b = alpha;
        double *C = &diag[0];

        C[0] = param.Cn;
        C[1] = 0;
        C[2] = param.Cp;

        w_index.clear();
        for(size_t j = 0; j < w_size; j++) {
            w_index.push_back(j);
            w[j] = 0;
            xj_sq[j] = 0;
        }

        if(param.bias > 0) {
            bb = 0;
            w_index.push_back(w_size);
            xj_sq[w_size] = 0;
            bias_column_vec.nnz = index.size();
            for(size_t idx = 0; idx < index.size(); idx++) {
                bias_column_vec.idx[idx] = index[idx];
                bias_column_vec.val[idx] = param.bias;
                xj_sq[w_size] += C[GETI(index[idx])] * param.bias * param.bias;
            }
            std::sort(bias_column_vec.idx, bias_column_vec.idx + index.size());
        }

        for(auto& i : index) {
            const svec_t& xi = X.get_row(i);
            b[i] = 1.0 - y[i] * do_dot_product_with_bias(xi, param.bias, w, bb, param);
            for(size_t t = 0; t < xi.nnz; t++) {
                xj_sq[xi.idx[t]] += C[GETI(i)] * xi.val[t] * xi.val[t];
            }
        }

        double Gmax_new = 0, Gmax_old = INF;
        double Gnorm1_new = 0;
        double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration

        size_t active_size = w_index.size();
        size_t iter = 0;
        while(iter < param.max_iter) {
            Gmax_new = 0;
            Gnorm1_new = 0;

            // shuffle
            rng.shuffle(w_index.begin(), w_index.begin() + active_size);
            size_t s = 0;
            for(s = 0; s < active_size; s++) {
                size_t j = w_index[s];
                const svec_t& xj = (j == w_size)? bias_column_vec : X.get_col(j);
                ValueType& wj = (j == w_size)? bb : w[j];

                double G_loss = 0; // gradient of loss term
                double H = 0;      // hessian

                for(size_t t = 0; t < xj.nnz; t++) {
                    size_t i = xj.idx[t];
                    if(b[i] > 0) {
                        double val = y[i] * xj.val[t];
                        double tmp = C[GETI(i)] * val;
                        G_loss -= tmp * b[i];
                        H += tmp * val;
                    }
                }
                G_loss *= 2;

                H *= 2;
                H = std::max(H, 1e-12);

                double Gp = G_loss + 1;
                double Gn = G_loss - 1;
                double violation = 0;
                if(wj == 0) {
                    if(Gp < 0) {
                        violation = -Gp;
                    } else if(Gn > 0) {
                        violation = Gn;
                    } else if(Gp > (Gmax_old / index.size()) && Gn < (-Gmax_old / index.size())) {
                        active_size--;
                        std::swap(w_index[s], w_index[active_size]);
                        s--;
                        continue;
                    }
                } else if(wj > 0) {
                    violation = fabs(Gp);
                } else { // wj < 0
                    violation = fabs(Gn);
                }

                Gmax_new = std::max(Gmax_new, violation);
                Gnorm1_new += violation;

                // obtain Newton direction d
                double d = 0;
                if(Gp < (H * wj)) {
                    d = -Gp / H;
                } else if(Gn > (H * wj)) {
                    d = -Gn / H;
                } else {
                    d = -wj;
                }
                if(fabs(d) < 1e-12) {
                    continue;
                }

                double delta = fabs(wj + d) - fabs(wj) + G_loss * d;
                double d_old = 0;
                double loss_old = 0, loss_new = 0;
                size_t num_linesearch = 0;
                for(num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
                    double d_diff = d_old - d;
                    double cond = fabs(wj + d) - fabs(wj) - sigma * delta;
                    double appxcond = xj_sq[j] * d * d + G_loss * d + cond;
                    if(appxcond <= 0) {
                        //do_axpy(d_diff, xj, b);
                        for(size_t t = 0; t < xj.nnz; t++) {
                            size_t i = xj.idx[t];
                            b[i] += d_diff * y[i] * xj.val[t];
                        }
                        break;
                    }

                    loss_new = 0;
                    for(size_t t = 0; t < xj.nnz; t++) {
                        size_t i = xj.idx[t];
                        double xij = y[i] * xj.val[t];
                        if(num_linesearch ==0) {
                            if(b[i] > 0) {
                                loss_old += C[GETI(i)] * b[i] * b[i];
                            }
                        }
                        double b_new = b[i] + d_diff * xij;
                        b[i] = b_new;
                        if(b_new > 0) {
                            loss_new += C[GETI(i)] * b_new * b_new;
                        }
                    }

                    cond = cond + loss_new - loss_old;
                    if(cond <= 0) {
                        break;
                    } else {
                        d_old = d;
                        d *= 0.5;
                        delta *= 0.5;
                    }
                }

                wj += d;

                // recompute b[] if line search takes too many steps
                if(num_linesearch >= max_num_linesearch) {
                    for(auto& i : index) {
                        const svec_t& xi = X.get_row(i);
                        b[i] = 1.0 - y[i] * do_dot_product_with_bias(xi, param.bias, w, bb, param);
                    }
                }
            }

            if(iter == 0) {
                Gnorm1_init = Gnorm1_new;
            }
            iter++;

            if(Gnorm1_new <= param.eps * Gnorm1_init) {
                if(active_size == w_index.size()) {
                    break;
                } else {
                    active_size = w_index.size();
                    Gmax_old = INF;
                    continue;
                }
            }

            Gmax_old = Gmax_new;
        }
	}

    double compute_obj() {
        // compute obj
        double obj = 0;
        if(param.solver_type == L2R_L1LOSS_SVC_DUAL ||
                param.solver_type == L2R_L2LOSS_SVC_DUAL) {
            double reg = do_dot_product(w, w);
            double v = 0;
            for(auto &i : index) {
                v += alpha[i] * (alpha[i] * diag[GETI(i)] - 2);
            }
            double obj = reg * 0.5 + v * 0.5;
            printf("obj %g |w| %g\n", obj, reg);
        } else if(param.solver_type == L2R_LR_DUAL) {
            double reg = do_dot_product(w, w);
            double v = 0;
            for(auto &i : index) {
                v += alpha[2 * i] * log(alpha[2 * i]) \
                     + alpha[2 * i + 1] * log(alpha[2 * i + 1]) \
                     - upper_bound[GETI(i)] * log(upper_bound[GETI(i)]);
            }
            double obj = reg * 0.5 + v;
            printf("obj %g |w| %g\n", obj, reg);
        } else if(param.solver_type == L1R_L2LOSS_SVC) {
            size_t nnz = 0;
            double reg = 0;
            for(size_t j = 0; j < w_size; j++) {
                if(w[j] != 0) {
                    nnz += 1;
                    reg += fabs(w[j]);
                }
            }
            const double *C = &diag[0];
            const dvec_t &b = alpha;
            double v = 0;
            for(auto &i : index) {
                v += C[GETI(i)] * ((b[i] > 0) ? b[i] * b[i] : 0);
            }
            obj = reg + v;
            printf("obj %g |w|(%ld, %g) |v| %g\n", obj, nnz, reg, v);
        }
        return obj;
    }
};
#undef GETI


// Z = Y * C
struct SVMJob {
    const gmat_wrapper_t* feat_mat; // m \times d
    const smat_t* Y;                // m \times n
    const smat_t* C;                // n \times k: NULL to denote the pure Multi-label setting
    const smat_t* Z;                // m \times k: NULL to denote the pure Multi-label setting
    size_t code;     // code idx in C (i.e., column index of C)
    size_t subcode;  // index of the label with code (i.e. column index of Y or row index of C)
    const SVMParameter *param_ptr;

    SVMJob(const gmat_wrapper_t *feat_mat, const smat_t *Y, const smat_t *C, const smat_t *Z,
            size_t code , size_t subcode, const SVMParameter *param_ptr=NULL):
        feat_mat(feat_mat), Y(Y), C(C), Z(Z),
        code(code), subcode(subcode), param_ptr(param_ptr){ }

    void init_worker(SVMWorker& worker) const {
        size_t w_size = feat_mat->cols();
        size_t y_size = feat_mat->rows();
        worker.lazy_init(w_size, y_size, param_ptr);
        worker.index.clear();
        if(Z != NULL) {
            // multilabel setting with codes for labels
            const svec_t& z_c = Z->get_col(code);
            for(size_t idx = 0; idx < z_c.nnz; idx++) {
                size_t i = z_c.idx[idx];
                worker.index.push_back(i);
                worker.y[i] = -1;
            }
        } else {
            // pure multilabel setting without additional codes
            for(size_t i = 0; i < y_size; i++) {
                worker.index.push_back(i);
                worker.y[i] = -1;
            }
        }
        const svec_t& y_s = Y->get_col(subcode);
        for(size_t idx = 0; idx < y_s.nnz; idx++) {
            size_t i = y_s.idx[idx];
            worker.y[i] = +1;
        }
    }

    void solve(SVMWorker& worker, coo_t& coo_model, double threshold=0.0) const {
        worker.solve(*feat_mat);
        for(size_t i = 0; i < worker.w_size; i++) {
            coo_model.push_back(i, subcode, worker.w[i], threshold);
        }
        if(param_ptr->bias > 0) {
            coo_model.push_back(worker.w_size, subcode, worker.bb, threshold);
        }
    }

    void solve(SVMWorker& worker, dmat_t& W, double threshold=0.0) const {
        worker.solve(*feat_mat);
        for(size_t i = 0; i < worker.w_size; i++) {
            if(fabs(worker.w[i]) >= threshold) {
                W.at(i, subcode) = worker.w[i];
            }
        }
        if(param_ptr->bias > 0 && fabs(worker.bb) >= threshold) {
            W.at(worker.w_size, subcode) = worker.bb;
        }
    }

    void reset_worker(SVMWorker& worker) const {
        worker.index.clear();
        worker.bias_column_vec.nnz = 0;
        if(Z != NULL) {
            const svec_t& z_c = Z->get_col(code);
            for(size_t idx = 0; idx < z_c.nnz; idx++) {
                size_t i = z_c.idx[idx];
                worker.y[i] = 0;
            }
        } else {
            std::fill_n(worker.y.data(), feat_mat->rows(), 0);
        }
    }
};

void multilabel_train_with_codes(const gmat_wrapper_t *feat_mat, const smat_t *Y, const smat_t *C, const smat_t *Z, coo_t *model, double threshold, SVMParameter* param, int threads) {
    size_t w_size = feat_mat->cols();
    size_t y_size = feat_mat->rows();
    size_t nr_labels = Y->cols;

    std::vector<SVMWorker> worker_set(threads);
    std::vector<coo_t> model_set(threads);

#pragma omp parallel for schedule(static,1)
    for(int tid = 0; tid < threads; tid++) {
        worker_set[tid].init(w_size, y_size, param);
        model_set[tid].reshape(w_size + (param->bias > 0), nr_labels);
    }

    std::vector<SVMJob> job_queue;
    if(C != NULL && Z != NULL) {
        size_t code_size = C->cols;
        for(size_t code = 0; code < code_size; code++) {
            const svec_t& C_code = C->get_col(code);
            for(size_t idx = 0; idx < C_code.nnz; idx++) {
                size_t subcode = static_cast<size_t>(C_code.idx[idx]);
                job_queue.push_back(SVMJob(feat_mat, Y, C, Z, code, subcode, param));
            }
        }
    } else {
        // either C == NULL or Z == NULL
        // pure multilabel setting
        for(size_t subcode = 0; subcode < nr_labels; subcode++) {
            job_queue.push_back(SVMJob(feat_mat, Y, NULL, NULL, 0, subcode, param));
        }
    }
#pragma omp parallel for schedule(dynamic,1)
    for(size_t job_id = 0; job_id < job_queue.size(); job_id++) {
        int tid = omp_get_thread_num();
        SVMWorker& worker = worker_set[tid];
        coo_t& local_model = model_set[tid];
        const SVMJob& job = job_queue[job_id];
        job.init_worker(worker);
        job.solve(worker, local_model, threshold);
        job.reset_worker(worker);
    }
    model->reshape(w_size, nr_labels);
    model->swap(model_set[0]);
    for(int tid = 1; tid < threads; tid++) {
        model->extends(model_set[tid]);
    }
}

void multilabel_predict_with_pred_labels(const smat_t *feat_mat, const smat_t *W, const smat_t *csc_labels, ValueType *pred_values, int threads) {
    if(threads < 1) { threads = omp_get_num_procs(); }
    threads = std::min(threads, omp_get_num_procs());

    std::vector<sdvec_t> sdvec_set(threads);

    for(int tid = 0; tid < threads; tid++) {
        sdvec_set[tid].resize(W->rows);
    }

    if(csc_labels->cols <= static_cast<unsigned>(threads)) {
        for(size_t c = 0; c < csc_labels->cols; c++) {
            if(csc_labels->nnz_of_col(c) == 0) { continue; }
            sdvec_t* Wc = &sdvec_set[0];
            Wc->init_with_svec(W->get_col(c));
#pragma omp parallel for schedule(dynamic, 1) shared(feat_mat, csc_labels, pred_values, Wc)
            for(size_t idx = csc_labels->col_ptr[c]; idx < csc_labels->col_ptr[c + 1]; idx++) {
                size_t r = csc_labels->row_idx[idx];
                pred_values[idx] = do_dot_product(feat_mat->get_row(r), *Wc);
            }
        }
    } else {
#pragma omp parallel for schedule(dynamic, 1) shared(feat_mat, csc_labels, pred_values)
        for(size_t c = 0; c < csc_labels->cols; c++) {
            if(csc_labels->nnz_of_col(c) == 0) { continue; }
            int tid = omp_get_thread_num();
            sdvec_t& Wc = sdvec_set[tid];
            Wc.init_with_svec(W->get_col(c));
            for(size_t idx = csc_labels->col_ptr[c]; idx != csc_labels->col_ptr[c + 1]; idx++) {
                size_t r = csc_labels->row_idx[idx];
                pred_values[idx] = do_dot_product(feat_mat->get_row(r), Wc);
            }
        }
    }
}

void load_label_from_svmlight(char *filename, size_t nr_lines, dvec_t& y) {
    y.resize(nr_lines);

    std::string line, label;
    std::ifstream fs;
    fs.open(filename, std::ios::in);
    if(!fs.is_open()) {
        std::cout << "Unable to open" << filename << std::endl;
        exit(-1);
    }
    size_t line_num = 0;
    while(std::getline(fs, line)) {
        if(fs.eof()) {
            break;
        }
        y[line_num] = static_cast<ValueType>(strtod(line.c_str(), NULL));
        line_num += 1;
    }
}


// ====================== Python/Ctypes Interface ================================
#ifdef __cplusplus
extern "C" {
#endif

void get_codes(const PyMatrix* py_mat_ptr, uint32_t depth, uint32_t partition_algo, int seed, uint32_t max_iter, int threads, uint32_t* label_codes) {
    //fprintf(stderr, "threads = %d %d\n", threads, omp_get_num_procs());
    //fprintf(stderr, "algo = %d\n", partition_algo);
    if(threads < 1) { threads = omp_get_num_procs(); }
    threads = std::min(threads, omp_get_num_procs());
    //fprintf(stderr, "threads = %d\n", threads);
    omp_set_num_threads(threads);
    const gmat_wrapper_t feat_mat(py_mat_ptr);
    Tree tree(depth);
    tree.run_clustering(feat_mat, partition_algo, seed, label_codes, max_iter);
}

void c_sparse_inner_products(const PyMatrix *pX, const PyMatrix *pM, uint64_t len, uint32_t *X_row_idx, uint32_t *M_col_idx, void* val, int threads) {
    if(threads < 1) { threads = omp_get_num_procs(); }
    threads = std::min(threads, omp_get_num_procs());
    omp_set_num_threads(threads);

    const gmat_wrapper_t X(pX);
    const gmat_wrapper_t M(pM);
    compute_sparse_entries_from_gmat_x_gmat(X.get_gmat(), M.get_gmat(), len, X_row_idx, M_col_idx, static_cast<ValueType*>(val));
    /*
    if(X.is_sparse() && M.is_sparse()) {
        const smat_t& sX = X.get_sparse();
        const smat_t& sM = M.get_sparse();
#pragma omp parallel for schedule(dynamic,64)
        for(size_t idx = 0; idx < len; idx++) {
            const svec_t& xi = sX.get_row(X_row_idx[idx]);
            const svec_t& mj = sM.get_col(M_col_idx[idx]);
            val[idx] = static_cast<float>(do_dot_product(xi, mj));
        }
    } else if(X.is_sparse() && M.is_dense()) {
        const smat_t& sX = X.get_sparse();
        const dmat_t& dM = M.get_dense();
#pragma omp parallel for schedule(dynamic,64)
        for(size_t idx = 0; idx < len; idx++) {
            const svec_t& xi = sX.get_row(X_row_idx[idx]);
            uint32_t j = M_col_idx[idx];
            double tmp = 0;
            for(size_t t = 0; t < xi.nnz; t++) {
                tmp += xi.val[t] * dM.at(xi.idx[t], j);
            }
            val[idx] = tmp;
        }
    } else if(X.is_dense() && M.is_sparse()) {
        const dmat_t& dX = X.get_dense();
        const smat_t& sM = M.get_sparse();
#pragma omp parallel for schedule(dynamic,64)
        for(size_t idx = 0; idx < len; idx++) {
            const svec_t& mj = sM.get_col(M_col_idx[idx]);
            uint32_t i = X_row_idx[idx];
            double tmp = 0;
            for(size_t t = 0; t < mj.nnz; t++) {
                tmp += mj.val[t] * dX.at(i, mj.idx[t]);
            }
            val[idx] = tmp;
        }
    } else if(X.is_dense() && M.is_dense()) {
        const dmat_t& dX = X.get_dense();
        const dmat_t& dM = M.get_dense();
#pragma omp parallel for schedule(static,64)
        for(size_t idx = 0; idx < len; idx++) {
            uint32_t i = X_row_idx[idx];
            uint32_t j = M_col_idx[idx];
            double tmp = 0;
            for(size_t t = 0; t < dX.cols; t++) {
                tmp += dX.at(i, t) * dM.at(t, j);
            }
            val[idx] = tmp;
        }
    }
    */
}

void c_multilabel_train_with_codes(
        const PyMatrix *pX,
        const PyMatrix *pY,
        const PyMatrix *pC,
        const PyMatrix *pZ,
        py_coo_allocator_t coo_alloc,
        double threshold,
        int solver_type,
        double Cp,
        double Cn,
        size_t max_iter,
        double eps,
        double bias,
        int threads) {

    if(threads == -1) {
        threads = omp_get_num_procs();
    }
    threads = std::min(threads, omp_get_num_procs());
    omp_set_num_threads(threads);

    const gmat_wrapper_t feat_mat(pX);
    const gmat_wrapper_t Y(pY);
    SVMParameter param(solver_type, Cp, Cn, max_iter, eps, bias);
    coo_t model;
    if(pC != NULL && pZ != NULL) {
        const gmat_wrapper_t C(pC);
        const gmat_wrapper_t Z(pZ);
        multilabel_train_with_codes(
            &feat_mat,
            &Y.get_sparse(),
            &C.get_sparse(),
            &Z.get_sparse(),
            &model,
            threshold,
            &param,
            threads
        );
    } else {
        multilabel_train_with_codes(
            &feat_mat,
            &Y.get_sparse(),
            NULL,
            NULL,
            &model,
            threshold,
            &param,
            threads
        );
    }
    model.create_pycoo(coo_alloc);
}

void c_multilabel_predict_with_codes(
        const PyMatrix *pX,
        const PyMatrix *pW,
        const PyMatrix *pC,
        const PyMatrix *pZ, // csr_codes
        py_pred_allocator_t pred_alloc,
        int threads) {

    if(threads == -1) {
        threads = omp_get_num_procs();
    }
    threads = std::min(threads, omp_get_num_procs());
    omp_set_num_threads(threads);
    const gmat_wrapper_t X(pX);
    const gmat_wrapper_t W(pW);
    smat_t pred_labels;
    std::vector<ValueType> pred_values;
    if(pC != NULL && pZ != NULL) {
        const gmat_wrapper_t C(pC);
        const gmat_wrapper_t Z(pZ);
        smat_x_smat(Z.get_sparse(), C.get_sparse().transpose(), pred_labels);
        pred_values.resize(pred_labels.nnz, 0);
        multilabel_predict_with_pred_labels(
            &X.get_sparse(),
            &W.get_sparse(),
            &pred_labels,
            pred_values.data(),
            threads
        );
    }
    uint64_t* col_ptr=NULL;
    uint64_t* row_idx=NULL;
    ValueType* val1=NULL;
    ValueType* val2=NULL;

    pred_alloc(pred_labels.rows, pred_labels.cols, pred_labels.nnz,
            &col_ptr, &row_idx, &val1, &val2);

#pragma omp parallel for
    for(size_t c = 0; c <= pred_labels.cols; c++) {
        col_ptr[c] = pred_labels.col_ptr[c];
    }

#pragma omp parallel for
    for(size_t idx = 0; idx < pred_labels.nnz; idx++) {
        row_idx[idx] = pred_labels.row_idx[idx];
        val1[idx] = pred_labels.val[idx];
        val2[idx] = pred_values[idx];
    }
}

#ifdef __cplusplus
} // end of extern "C"
#endif

int main(int argc, char* argv[]) {
    char *filename = argv[1];
    int solver_type = atoi(argv[2]);
    size_t nr_skips = 1;
    bool zero_based = false;
    double append_bias = -1;
    smat_t X;
    X.load_from_svmlight(filename, nr_skips, zero_based, append_bias);
    SVMWorker ws;
    SVMParameter param;
    param.solver_type = solver_type;
    size_t w_size = X.cols, y_size = X.rows;
    ws.init(w_size, y_size, &param);
    load_label_from_svmlight(filename, X.rows, ws.y);

    printf("nnz %ld, rows %ld, col %ld\n",
            X.get_nnz(),
            X.rows, X.cols);
    // classification
    ws.index.clear();
    for(size_t i = 0; i < X.rows; i++) {
        ws.index.push_back(i);
    }

    ws.solve(X);
    ws.compute_obj();
    return 0;
}
