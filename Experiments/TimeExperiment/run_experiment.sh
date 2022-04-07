#!/bin/bash -l

#### LargeBrainAtlas dataset ----------------------------------------------
declare -a embedding_size_list=("20" "100" "500")

logdirbase='out/siVAE/LargeBrain'

for embedding_size in "${embedding_size_list[@]}"
do

  ## Use all feature attribution methods only for the smallest embedding size
  if [ "$embedding_size" == "20" ]; then
    FA_method="All"
  else
    FA_method="Subset"
  fi

  ## Run siVAE
  python repo_FetalLiver_3_model.py --method "siVAE" \
                                    --do_FA "True" \
                                    --logdir "${logdirbase}/${embedding_size}" \
                                    --num_reduced "5000" \
                                    --reduce_mode "PCA" \
                                    --dataset "LargeBrain" \
                                    --FA_method "${FA_method}" \
                                    --embedding_size "${embedding_size}"
done

#### BrainCortex dataset ----------------------------------------------

declare -a embedding_size_list=("120" "100" "500")

logdirbase='out/siVAE/BrainCortex'

for embedding_size in "${embedding_size_list[@]}"
do

  FA_method="Subset"

  ## Run siVAE
  python repo_FetalLiver_3_model.py --method "siVAE" \
                                    --do_FA "True" \
                                    --logdir "${logdirbase}/${embedding_size}" \
                                    --dataset "BrainCortex" \
                                    --FA_method "${FA_method}" \
                                    --embedding_size "${embedding_size}"
done
