#!/bin/bash

AGENT=oracle_agent
DATA_DIR=data2

frames=$1
if [ -z $frames ]; then
    echo "usage: $0 <frames>"
    exit 1
fi

rm -rf $DATA_DIR*
python3 -m tournament.play -s $DATA_DIR -f $1 AI $AGENT AI
