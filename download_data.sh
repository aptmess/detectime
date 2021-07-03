#!/bin/sh
# brew install wget
mkdir data/INPUT_DATA
# shellcheck disable=SC2164
cd data/INPUT_DATA
wget -i ../../download.txt

echo 'start unzip train_data* folders'
# mac os
find . -name 'train_data*.zip' -exec sh -c 'unzip -d "${1%.*}" "$1"' _ {} \;
# ubuntu
# find -name 'train_data*.zip' -exec sh -c 'unzip -d "${1%.*}" "$1"' _ {} \;

echo 'done'

echo 'start moving pictures into folder TRAIN_DATA/'

find . -type d -name 'train_data*' -exec sh -c 'mv {}/* TRAIN_DATA/' \;
find . -type d -name 'train_data*' -exec sh -c 'rm -r {}' \;

echo 'done'

echo 'start moving zip archives into zip directory'

mkdir ZIP/
find . -name 'train_data*.zip' -exec mv {} ZIP/ \;

echo 'done'