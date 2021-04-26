CXX=g++
#CXXFLAGS=-fopenmp -static -O3
CXXFLAGS=-fopenmp -ffast-math -pipe -g -fPIC
CXXFLAGS=-fopenmp -march=native -ffast-math -fPIC -g -std=c++11
#CXXFLAGS=-fopenmp -march=native -ffast-math -O3 -std=c++11
CXXFLAGS=-fopenmp -march=native -ffast-math -std=c++11 -O3
CXXFLAGS=-fopenmp -march=native -O3 -std=c++11 -fPIC
CXXFLAGS_DEBUG=-fopenmp -march=native -g -std=c++11 -fPIC -pg
VERSION=0.1

#CXXFLAGS+= -D_USE_FLOAT_

SCIPY_ROOT=$(shell python -c "from scipy import linalg as lib;print(lib.__file__.rsplit('/', 1)[0])")
SCIPY_BLAS=$(shell python -c "from scipy.linalg import cython_blas as lib;print(lib.__file__)")
SCIPY_LAPACK=$(shell python -c "from scipy.linalg import cython_lapack as lib;print(lib.__file__)")

SCIPY_BLAS=$(shell python -W ignore -c "import numpy.distutils.system_info as C;print(' '.join('-l{}'.format(x) for x in C.get_info('lapack_opt')['libraries']))" | tail -n 1)
SCIPY_BLAS_DIR=$(shell python -W ignore -c "import numpy.distutils.system_info as C;print(' '.join('-L{}'.format(x) for x in C.get_info('lapack_opt')['library_dirs']))" | tail -n 1)
SCIPY_BLAS_RPATH=$(shell python -W ignore -c "import numpy.distutils.system_info as C;print('-Wl,-rpath,{}'.format(':'.join(C.get_info('lapack_opt')['library_dirs'])))" | tail -n 1)

BLASFLAGS=-llapack_atlas -lf77blas -lcblas -latlas -lgfortran -L/u/rofuyu/.local/lib  #GCC in UTCS
BLASFLAGS=-llapack -lptf77blas -lptcblas -latlas -L/home/01845/rofuyu/.local/lib  # GCC in TACC
BLASFLAGS=-lblas -llapack
BLASFLAGS=-lmklml_intel
LIBSFLAGS=-L/home/rofuyu/anaconda3/lib -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp
#LIBSFLAGS=-L$(SCIPY_ROOT) $(SCIPY_BLAS) $(SCIPY_LAPACK)
BLASFLAGS=${LIBSFLAGS}
LIBSFLAGS=${SCIPY_BLAS_RPATH} ${SCIPY_BLAS_DIR} ${SCIPY_BLAS}

OS=$(shell uname)

all: lib go-linear

rf_linear.o: rf_linear.cpp rf_matrix.h
	${CXX} ${CXXFLAGS} -Wall -c -o rf_linear.o rf_linear.cpp

go-linear: rf_linear.o
	${CXX} ${CXXFLAGS_DEBUG} -DValueType=double -Wall -g -o go-linear rf_linear.o ${LIBSFLAGS}

lib: rf_linear_so

rf_linear_so: rf_linear.cpp rf_matrix.h
	SO=rf_linear; \
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,$${SO}"; \
		BLAS_LIB_FLAG="-framework Accelerate";\
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,$${SO}"; \
		BLAS_LIB_FLAG=" -Wl,--as-needed ${SCIPY_BLAS_RPATH} ${SCIPY_BLAS_DIR} ${SCIPY_BLAS} -liomp5";\
	fi; \
	echo CXXFLAGs=${CXXFLAGS}; \
	echo BLAS_LIB_FLAG=$${BLAS_LIB_FLAG}; \
	${CXX} ${CXXFLAGS} -D ValueType=float $${SHARED_LIB_FLAG} -o $${SO}_float32.so rf_linear.cpp $${BLAS_LIB_FLAG} ; \
	${CXX} ${CXXFLAGS} -D ValueType=double $${SHARED_LIB_FLAG} -o $${SO}_float64.so rf_linear.cpp $${BLAS_LIB_FLAG}; \

clean:
	rm -rf go-linear rf_linear*.so *.o

