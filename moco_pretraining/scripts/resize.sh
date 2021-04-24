#!/bin/bash
# Convert all images in $1 to $2

mkdir -p $2

for filename in $1/*; do
    # echo $filename
    convert $filename -resize 500x500! $2/$(basename "$filename")
done