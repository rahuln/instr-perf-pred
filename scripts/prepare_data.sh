#!/bin/bash

echo "Downloading Super-NaturalInstructions dataset..."
wget -P data/eval/superni/ https://github.com/allenai/natural-instructions/archive/refs/heads/master.zip
unzip data/eval/superni/master.zip -d data/eval/superni/ && rm data/eval/superni/master.zip
mv data/eval/superni/natural-instructions-master/* data/eval/superni/ && rm -r data/eval/superni/natural-instructions-master

echo "Preparing custom subset of the BIG-bench dataset..."
unzip data/eval/big-bench.zip -d data/eval/ && rm data/eval/big-bench.zip
