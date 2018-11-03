#!/bin/bash

# Set image name
IMAGE="cowc_counting:test"

# Get project root dicrectory
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# Buld docker image from project root directory
cd ${PROJ_DIR} && \
nvidia-docker build -t ${IMAGE} -f ${THIS_DIR}/Dockerfile .