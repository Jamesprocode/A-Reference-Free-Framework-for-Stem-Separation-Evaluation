#!/bin/bash

set -euo pipefail

for i in {1..5}; do
    make 
    convert -verbose -density 150 -trim document.pdf[2] -quality 100 -flatten -sharpen 0x1.0 recursion.png
done
