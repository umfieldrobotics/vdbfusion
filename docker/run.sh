#!/usr/bin/env bash
docker run -it \
    -e HOME \
    -v $HOME:/home/anjashep-frog-lab \
    -v /usr/local/lib/cmake:/usr/local/lib/cmake \
    -v /usr/local/include/openvdb:/usr/local/include/openvdb \
    -v /usr/local/include/nanovdb:/usr/local/include/nanovdb \
    --name vdbfusion_docker \
    --gpus all \
    vdbfusion_docker # TODO: change this image name, container name and home directory accordingly
