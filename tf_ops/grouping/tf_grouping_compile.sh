#/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

/usr/local/cuda/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC \
  -I ${TF_INC} \
  -I ${TF_INC}/external/nsync/public \
  -I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ \
  -L${TF_LIB} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0 
