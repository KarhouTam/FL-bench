#!/bin/bash

if [ ! -d "../medmnistA/raw" ]; then
    mkdir -p ../medmnistA/raw
fi

cd ../medmnistA/raw

wget https://wjdcloud.blob.core.windows.net/dataset/cycfed/medmnistA.tar.gz

tar -xzvf medmnistA.tar.gz

mv medmnistA/* ./

rm -rf medmnistA

cd ../..
