# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#FROM ubuntu:20.04
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# setup environment
ENV TZ=US
ENV TERM xterm
ENV DEBIAN_FRONTEND=noninteractive

# Install essentials
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    ccache \
    clang-format \
    git \
    sudo \
    wget \
    tar \
    vim \
    libtiff-dev \
    libfontconfig1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install cmake
RUN mkdir ~/temp
RUN cd ~/temp
RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-x86_64.tar.gz
RUN tar xvzf cmake-3.28.6-linux-x86_64.tar.gz --strip-components=1 -C /usr/local
RUN cd ..
RUN rm -rf temp

# Install Python apt dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3 \
    python3-numpy \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python pip dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade \
    black \
    numpy \
    pytest \
    setuptools \
    twine \
    wheel

# Install C++ Dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    libblosc-dev \
    libboost-iostreams-dev \
    libboost-numpy-dev \
    libboost-python-dev \
    libboost-system-dev \
    libeigen3-dev \
    libtbb-dev \
    libc++-dev \
    libglu1-mesa-dev \
    python-dev \
    python-numpy \
    && rm -rf /var/lib/apt/lists/*

# Install opencv
WORKDIR /
RUN apt-get update && apt-get install -y wget git ssh libfreetype6 libfreetype6-dev libharfbuzz-dev
RUN git clone https://github.com/opencv/opencv_contrib.git --branch 4.6.0
RUN git clone https://github.com/opencv/opencv.git --branch 4.6.0
RUN cd /opencv
RUN mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_FREETYPE=OFF \
      ../opencv && \
    make -j6 && \
    make install

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

ENV TERM xterm-256color
CMD ["bash", "-l"]

USER docker
RUN export PATH="/usr/local/cuda/bin:PATH"
CMD /bin/bash

