#!/usr/bin/env bash

docker run -it \
    -e HOME \
    -v /home/yuzhen/Desktop/semanticVDB:/home/yuzhen/Desktop/semanticVDB \
    -v /usr/local/lib/cmake/OpenVDB:/usr/local/lib/cmake/OpenVDB \
    -v /usr/local/lib/cmake/VDBFusion:/usr/local/lib/cmake/VDBFusion \
    --name vdbfusion_docker \
    --gpus all \
    vdbfusion_docker # TODO: change this image name, container name and home directory accordingly
