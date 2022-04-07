## From https://zenodo.org/record/4333872#.YbEwJS-B2gQ

## D11, D30, D52, combined sampled
wget --content-disposition https://zenodo.org/record/4333872/files/D11.h5?download=1
wget --content-disposition https://zenodo.org/record/4333872/files/D30.h5?download=1
wget --content-disposition https://zenodo.org/record/4333872/files/D52.h5?download=1
wget --content-disposition https://zenodo.org/record/4333872/files/all_timepoints_subsampled.h5?download=1

wget --content-disposition https://zenodo.org/record/4333872/files/coloc_gtex_25_traits.tsv.gz?download=1
wget --content-disposition https://zenodo.org/record/4333872/files/coloc_neuroseq_25_traits.tsv.gz?download=1

gunzip coloc_gtex_25_traits.tsv.gz
gunzip coloc_neuroseq_25_traits.tsv.gz
