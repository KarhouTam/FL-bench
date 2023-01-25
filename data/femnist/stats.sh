#!/bin/bash


NAME="femnist"

cd ../leaf_utils

python3 stats.py --name $NAME

cd ../$NAME