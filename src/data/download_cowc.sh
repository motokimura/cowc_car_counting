#!/bin/bash

URL=ftp://gdo152.ucllnl.org/cowc/datasets/ground_truth_sets/

if [ $# -eq 1 ]; then
    DST_DIR=$1
else
	THIS_DIR=$(cd $(dirname $0); pwd)
	PROJ_DIR=`dirname ${THIS_DIR}`
	PROJ_DIR=`dirname ${PROJ_DIR}`
	DST_DIR=${PROJ_DIR}/data
fi

mkdir -p ${DST_DIR}

wget -r -nH ${URL} -P ${DST_DIR}
