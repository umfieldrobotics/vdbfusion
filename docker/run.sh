#!/usr/bin/env bash

docker run -it \
    -e HOME \
    -v $HOME:/home/anjashep-frog-lab \
    -v /usr/local/lib/cmake:/usr/local/lib/cmake \
    -v /usr/local/include/openvdb:/usr/local/include/openvdb \
    -v /usr/local/include/nanovdb:/usr/local/include/nanovdb \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name vdbfusion_docker \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    vdbfusion_docker # TODO: change this image name, container name and home directory accordingly
