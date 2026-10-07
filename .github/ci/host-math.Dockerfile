FROM buildpack-deps:noble-curl@sha256:48a80dfb7e2f2d1e93dbfff934fd2f187f306eef36cb5359ef8bba0ffe9205b3

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates cmake ninja-build g++ ccache \
      libeigen3-dev libopencv-dev libopencv-contrib-dev \
      libboost-dev libboost-system-dev libboost-filesystem-dev \
      libboost-thread-dev libboost-date-time-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CCACHE_DIR=/work/.ci-cache/ccache \
    CCACHE_BASEDIR=/work \
    CCACHE_COMPILERCHECK=content \
    CCACHE_MAXSIZE=1G \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /work
