#!/bin/bash -l

## =============================================================================
## 1. Preprocess data
## =============================================================================

python repo_FetalLiver_1-preprocess.py

## ===========================W
## =============================================================================

## The script is used by repo_FetalLiver_3_model.py
# python FetalLiver_2-SetupInput.py

## =============================================================================
## 3. Run model and their variantsW
## =============================================================================

#### siVAE models with variants ------------------------------------------------

## siVAE models
python repo_FetalLiver_3_model.py --method "siVAE" \
                                  --do_FA "True" \
                                  --logdir 'out/siVAE'

python repo_FetalLiver_3_model.py --method "siVAE-0" \
                                  --do_FA "True" \
                                  --logdir 'out/siVAE-0'

python repo_FetalLiver_3_model.py --method "siVAE-NB" \
                                  --do_FA "True" \
                                  --logdir 'out/siVAE-NB'

python repo_FetalLiver_3_model.py --method "siVAE-linear" \
                                  --do_FA "True" \
                                  --logdir 'out/siVAE-linear'

python repo_FetalLiver_3_model.py --method "siVAE-linear-NB" \
                                  --do_FA "True" \
                                  --logdir 'out/siVAE-linear-NB'

# ## VAE models
# python repo_FetalLiver_3_model.py --method "VAE" \
#                                   --do_FA "False" \
#                                   --logdir 'out/VAE'
#
# python repo_FetalLiver_3_model.py --method "VAE-linear" \
#                                   --do_FA "False" \
#                                   --logdir 'out/VAE-linear'

##scVI models
python repo_FetalLiver_3_model.py --method "scVI" \
                                  --do_FA "True" \
                                  --logdir 'out/scVI'

python repo_FetalLiver_3_model.py --method "LDVAE" \
                                  --do_FA "True" \
                                  --logdir 'out/LDVAE'

#### siVAE models with downsample ----------------------------------------------
# declare -a number_reduced_list=("100" "1000" "10000" "20000")
declare -a reduce_mode_list=("sample" "PCA")

logdirbase='out/siVAE_reduce'

## Test the accuracies with varying mode and num_reduced
for reduce_mode in "${reduce_mode_list[@]}"
do
  ## Set one base data dir for all runs for consistent train/test indexing
  if [ "$reduce_mode" == "${reduce_mode_list[0]}" ]; then
    datadirbase="${logdirbase}/${reduce_mode}-${number_reduced_list[0]}";
  fi
  ## Set num_sample_reduced for different modes
  if [ "$reduce_mode" == "PCA" ]; then
    declare -a number_reduced_list=("1" "10" "50" "500" "1000")
  elif [ "$reduce_mode" == "sample" ]; then
    declare -a number_reduced_list=("50" "500" "5000" "20000")
  fi
  ## Run siVAE
  for number_reduced in "${number_reduced_list[@]}"
  do
    python repo_FetalLiver_3_model.py --method "siVAE" \
                                      --do_FA "True" \
                                      --logdir "${logdirbase}/${reduce_mode}-${number_reduced}" \
                                      --num_reduced "${number_reduced}" \
                                      --reduce_mode "${reduce_mode}" \
                                      --datadir "${logdirbase}/${reduce_mode}-${number_reduced}/data_dict.pickle" \
                                      --datadirbase "${datadirbase}/data_dict.pickle"
  done
done

## Test extreme case where only few cells are kept for a single cell type

reduce_mode="sample"
# declare -a number_reduced_list=("10")
# declare -a number_reduced_list=("5" "50")
declare -a number_reduced_list=("1" "5" "50")
declare -a celltype_subset_list=("Kupffer_Cells" "Hepatocyte" "MHC_II_pos_B" "NK_NKT_cells")
# declare -a celltype_subset_list=("Hepatocytes")
# declare -a celltype_subset_list=("MHC_II_pos_B" "NK_NKT_cells")
# declare -a celltype_subset_list=("MHC_II_pos_B")

for ct_subset in "${celltype_subset_list[@]}"
do
  for number_reduced in "${number_reduced_list[@]}"
  do
    python repo_FetalLiver_3_model.py --method "siVAE" \
                                      --do_FA "True" \
                                      --logdir "${logdirbase}/${reduce_mode}-${number_reduced}-${ct_subset}" \
                                      --num_reduced "${number_reduced}" \
                                      --reduce_mode "${reduce_mode}" \
                                      --datadir "${logdirbase}/${reduce_mode}-${number_reduced}-${ct_subset}/data_dict.pickle" \
                                      --datadirbase "${datadirbase}/data_dict.pickle" \
                                      --reduce_subset "${ct_subset}"
  done
done


## =============================================================================
## 4. Perform GRN + gene relevance
## =============================================================================

## Prepare expression data for both original/siVAE-based GRN approaches
python prepare_input_for_grn.py

## Perform GRN
Rscript MI.R
python grnboost2.py

## Perform gene_relevance on siVAE
Rscript gene_relevance.R

## =============================================================================
## 5. Perform GroundTruthDegreeCentralityPrediction=
## =============================================================================

## Increase total number of open files to avoid error
ulimit -n 2048

python repo_FetalLiver_3_model.py --method "DegreeCentralityPrediction" \
                                  --do_FA "False" \
                                  --logdir 'out/DegreeCentralityPrediction'

## =============================================================================
## 6. Perform GroundTruthDegreeCentralityPrediction=
## =============================================================================

python repo_FetalLiver_4_analysis.py

## =============================================================================
## 7. Perform LE_dim testing
## =============================================================================

# declare -a reduce_mode_list=("1" "2" "3" "5" "10")
declare -a LE_dim_list=("1" "2" "3" "5" "10" "20")
declare -a method_list=("siVAE" "siVAE-0")
logdir_ledim='out/LE_dim'

for LE_dim in "${LE_dim_list[@]}"
do
  for method in "${method_list[@]}"
  do

    python repo_FetalLiver_3_model.py --method "${method}" \
                                      --do_FA "False" \
                                      --logdir "${logdir_ledim}/${method}/${LE_dim}" \
                                      --LE_dim "${LE_dim}" \
                                      --k_split "5" \
                                      --datadir "${logdir_ledim}/data_dict.pickle" \

  done
done

python repo_FetalLiver_3_model.py --method "siVAE-0" \
                                  --do_FA "False" \
                                  --logdir "out/betatest" \
                                  --LE_dim "20" \
                                  --k_split "5" \
                                  --datadir "${logdir_ledim}/data_dict.pickle" \


## =============================================================================
## 8. Perform gamma testing
## =============================================================================

# declare -a reduce_mode_list=("1" "2" "3" "5" "10")
declare -a scales_list=("0" "0.01" "0.05" "0.1" "0.5" "1" "10" "100")
declare -a method_list=("siVAE")
logdir_ledim='out/zv_recon_scale'

for method in "${method_list[@]}"
do
  for scale in "${scales_list[@]}"
  do

    python repo_FetalLiver_3_model.py --method "${method}" \
                                      --do_FA "False" \
                                      --logdir "${logdir_ledim}/${method}/${scale}" \
                                      --LE_dim "2" \
                                      --k_split "0.8" \
                                      --datadir "${logdir_ledim}/data_dict.pickle" \
                                      --zv_recon_scale "${scale}"
  done
done
