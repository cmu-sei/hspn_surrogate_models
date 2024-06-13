#!/bin/sh
cd /p/home/jyuko/projects/hspn_surrogate_models
python3 -m pip install -e .
#ls -lh /p/home/jyuko/.local/lib/python3.8/site-packages/
echo "python3   pyscripts/train_hspn.py  -a configs/don_volume_aoa.yml"
echo "-----"
python3 pyscripts/train_hspn.py  -a configs/don_volume_aoa.yml