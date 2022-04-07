
# Download data from zenodo

bash get_data.sh

# Process data into anndata for siVAE

taskset -a -c 0-19 python preprocess_D11.py
python preprocess_D52.py

# Optional lines to move data
get -r /share/quonlab/workspaces/yonchoi/data/iPSC_neuronal/data/rita_preprocessed/h5ad/* /home/yongin/projects/pypi/siVAE/Experiments/graph_comparison/data/iPSC_neuronal/
get -r /share/quonlab/workspaces/yonchoi/data/iPSC_neuronal/data/rita_preprocessed/h5ad/D11/experiment/P_FPP* /home/yongin/projects/pypi/siVAE/Experiments/graph_comparison/data/iPSC_neuronal/D11/experiment/P_FPP/

get -r /share/quonlab/workspaces/yonchoi/data/iPSC_neuronal/diff_efficiency_neur.csv /home/yongin/projects/pypi/siVAE/Experiments/graph_comparison/data/iPSC_neuronal/
get -r /share/quonlab/workspaces/yonchoi/data/iPSC_neuronal/data/D11/D11_scaled_sampled.h5ad /home/yongin/projects/pypi/siVAE/Experiments/graph_comparison/data/iPSC_neuronal/

# Run siVAE

# exp_list=("FPP" "P_FPP")
# exp_list=("P_FPP" "FPP")
# exp_list=("P_FPP")
exp_list=("FPP")

for exp in "${exp_list[@]}"
do

  downsample="500"
  logdir_exp="data/iPSC_neuronal/D11/experiment/${exp}"

  # Take union of cell lines selected for each efficiency and write to task.txt
  # cat "${logdir_exp}/lines-DA_efficiency.txt" \
  #     "${logdir_exp}/lines-diff_efficiency.txt" \
  #     "${logdir_exp}/lines-Sert_efficiency.txt" \
  #   | sort | uniq -u > "${logdir_exp}/task.txt"

  # For run all
  # find $logdir_exp -maxdepth 1 -mindepth 1 -type d -printf '%f\n' > "${logdir_exp}/task.txt"
  # taskfile="task.txt"

  # for downsampling
  python generate_run_list.py --datadir "${logdir_exp}" \
                              --size_threshold "${downsample}"
  taskfile='lines-downsampled.txt'

  for task in $(cat "${logdir_exp}/${taskfile}")
  do

    logdir="out/exp3/iPSC_neuronal/${downsample}/${exp}/${task}"
    rm -r $logdir ||: # Over write previous results
    datadir="${logdir_exp}/${task}/scaled_data.h5ad"
    indexdir="${logdir_exp}/${task}/kfold-index.npy"
    python run_siVAE.py --method "siVAE" \
                        --do_FA "False" \
                        --logdir "${logdir}" \
                        --LE_dim "32" \
                        --k_split "1" \
                        --datadir "${datadir}" \
                        --lr "0.0005" \
                        --zv_recon_scale "0.05" \
                        --iter "2000" \
                        --l2_scale "0.0001" \
                        --mb_size "128" \
                        --index_dir "${indexdir}"

  done
done

#### Run batch experiment
batch_list=("True")

datadir='data/iPSC_neuronal/D11_scaled_sampled.h5ad'

for batch in "${batch_list[@]}"
do

  python run_siVAE.py --method "siVAE" \
                      --do_FA "False" \
                      --logdir "out/exp2/iPSC_neuronal/batch/128/${batch}" \
                      --LE_dim "128" \
                      --k_split "1" \
                      --datadir "${datadir}" \
                      --lr "0.0005" \
                      --zv_recon_scale "0.00" \
                      --iter "2000" \
                      --l2_scale "0.0001" \
                      --mb_size "128" \
                      --use_batch "${batch}"

done


exp="FPP"
# task="HPSI0215i-yoch_6"
# task="HPSI0514i-fiaj_1"
task="HPSI0115i-hecn_6"
declare -a LE_dim_list=("128")
declare -a reduce_mode_list=("sample" "PCA")
declare -a activation_list=("elu" "relu")

for activation in "${activation_list[@]}"
do

  for reduce_mode in "${reduce_mode_list[@]}"
  do

    for LE_dim in "${LE_dim_list[@]}"
    do

      if [ "$reduce_mode" == "PCA" ]; then
        declare -a number_reduced_list=("1" "10" "50" "500" "1000")
      elif [ "$reduce_mode" == "sample" ]; then
        declare -a number_reduced_list=("50" "500" "5000" "20000")
      fi

      for number_reduced in "${number_reduced_list[@]}"
      do

        logdir="out/exp_test/iPSC_neuronal/test1/${exp}/${task}"
        logdir+="/${LE_dim}/${activation}/${reduce_mode}_${number_reduced}"

        rm -r $logdir ||:
        logdir_exp="data/iPSC_neuronal/D11/experiment/${exp}"
        datadir="${logdir_exp}/${task}/scaled_data.h5ad"

        python run_siVAE.py --method "siVAE" \
                            --do_FA "False" \
                            --logdir "${logdir}" \
                            --LE_dim "${LE_dim}" \
                            --k_split "1" \
                            --datadir "${datadir}" \
                            --lr "0.0005" \
                            --zv_recon_scale "0.00" \
                            --iter "2000" \
                            --l2_scale "0.0001" \
                            --mb_size "128" \
                            --activation_fun ${activation} \
                            --num_reduced "${number_reduced}" \
                            --reduce_mode "${reduce_mode}"
      done

    done

  done

done
