#!/bin/bash

if [ ! -d "../medmnistS/raw" ]; then
    mkdir -p ../medmnistS/raw
fi

cd ../medmnistS/raw

wget https://wjdcloud.blob.core.windows.net/dataset/cycfed/medmnist.tar.gz

tar -xzvf medmnist.tar.gz

mv medmnist/* ./

rm -rf medmnist
