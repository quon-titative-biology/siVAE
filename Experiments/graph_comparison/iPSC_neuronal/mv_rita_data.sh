mkdir rita_preprocessed
cp -r /share/quonlab/workspaces/ruoxinli/multitask_/data/iPSC_neuronal/Yongin/* rita_preprocessed/

python rda2h5ad.R
