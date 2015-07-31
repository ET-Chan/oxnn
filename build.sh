#!/bin/bash

cmake -E make_directory build && \
  cd build && \
  cmake .. && \
  make && \
  cp liboxnn.so ..
cd ..

# cd build && make install"
