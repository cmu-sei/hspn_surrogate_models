#!/bin/sh

# Default working directory
WORKDIR=$(pwd)

# Parse command line options
while getopts "w:" opt; do
  case $opt in
    w) WORKDIR=$OPTARG ;;
    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done

# Change to the specified working directory
cd $WORKDIR || { echo "Failed to change directory to $WORKDIR"; exit 1; }
echo "Changed working directory to $WORKDIR"



#python3 -m pip install --user -e .
#ls -lh /p/home/jyuko/.local/lib/python3.8/site-packages/
echo "python3   pyscripts/train_hspn.py  -a configs/don_volume_aoa.yml"
echo "-----"
python3 pyscripts/train_hspn.py  -a configs/don_volume_aoa.yml