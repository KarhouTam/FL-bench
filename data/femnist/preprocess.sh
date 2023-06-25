#!/bin/bash

# download data and convert to .json format

if [ ! -d "data/all_data" ] || [ ! "$(ls -A data/all_data)" ]; then
    cd preprocess
    bash data_to_json.sh
    cd ..
fi

NAME="femnist" # name of the dataset, equivalent to directory name

cd ../leaf_utils

bash preprocess.sh --name $NAME $@

cd ../$NAME

if [ -e "preprocess_args.json" ]; then
    rm preprocess_args.json
fi

touch preprocess_args.json

iu_default="0.01"
sf_default="0.1"
tf_default="0.9"

k=""
s=""
t=""
iu="$iu_default"
sf="$sf_default"
tf="$tf_default"
smplseed=$(sed -n '2p' meta/sampling_seed.txt)
spltseed=$(sed -n '2p' meta/split_seed.txt)

while [[ $# -gt 0 ]]; do
    case "$1" in
    -s)
        s="$2"
        shift
        ;;
    --iu)
        iu="$2"
        shift 2
        ;;
    --sf)
        sf="$2"
        shift 2
        ;;
    -k)
        k="$2"
        shift
        ;;
    -t)
        t="$2"
        shift
        ;;
    --tf)
        tf="$2"
        shift 2
        ;;
    *)
        shift
        ;;
    esac
done

args_json=$(
    cat <<EOF
{
  "s": "$s",
  "iu": "$iu",
  "sf": "$sf",
  "k": "$k",
  "t": "$t",
  "tf": "$tf",
  "smplseed": "$smplseed",
  "spltseed": "$spltseed"
}
EOF
)

echo "$args_json" >preprocess_args.json
