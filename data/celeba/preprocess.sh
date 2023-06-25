#!/usr/bin/env bash

# download data and convert to .json format

if [ ! -d "data/raw/img_align_celeba" ] || [ ! "$(ls -A data/raw/img_align_celeba)" ] || [ ! -f "data/raw/list_attr_celeba.txt" ]; then
	echo "Please download the celebrity faces dataset and attributes file from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
	exit 1
fi

if [ ! -f "data/raw/identity_CelebA.txt" ]; then
	echo "Please request the celebrity identities file from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
	exit 1
fi

if [ ! -d "data/all_data" ] || [ ! "$(ls -A data/all_data)" ]; then
	echo "Preprocessing raw data"
	python preprocess/metadata_to_json.py
fi

NAME="celeba" # name of the dataset, equivalent to directory name

cd ../leaf_utils

./preprocess.sh --name $NAME $@

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
