#!/usr/bin/env bash

docker run -it \
    -e HOME \
    -v $HOME:/home/anjashep-frog-lab \
    -v /usr/local/lib/cmake:/usr/local/lib/cmake \
    --name vdbfusion_docker \
    --gpus all \
    vdbfusion_docker # TODO: change this image name, container name and home directory accordingly
