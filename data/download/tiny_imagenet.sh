#!/bin/bash

if [ ! -d "../tiny_imagenet/raw" ]; then
    mkdir -p ../tiny_imagenet/raw
fi

cd ../tiny_imagenet/raw

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip

unzip tiny-imagenet-200.zip

mv tiny-imagenet-200/* ./

rm -rf tiny-imagenet-200

cd ../../
