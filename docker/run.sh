#!/usr/bin/env bash

docker run -it \
    -e HOME \
    -v $HOME:/home/anjashep-frog-lab \
    --name vdbfusion_docker \
    --gpus all \
    vdbfusion_docker # TODO: change this image name, container name and home directory accordingly
