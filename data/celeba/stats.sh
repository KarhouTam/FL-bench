#!/usr/bin/env bash

NAME="celeba"

cd ../leaf_utils

python3 stats.py --name $NAME

cd ../$NAME