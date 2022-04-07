#!/bin/bash

## Directories
logdir="/home/yongin/Gen/181130_IntMethod_test/1214/out/exp2_l2l1h1"

## Parameter Lists for Looping
# Network Setup
# declare -a network_setup_list=('0_5_5_5' '5_5_5' '0_5_5_15')
# declare -a network_setup_list=('0_5_5_15' '0_5_5_5' '5_5_5')
declare -a network_setup_list=('1.B-3-1_2.F-5')

# declare -a rho_list=('0.4' '0.6' '0.8')
declare -a rho_list=('0.6' '0.8')

# declare -a Z_dim_list=("3" "2")
declare -a Z_dim_list=("3" "2" "5")

declare -a h1_dim_list=("10")

# Hyperparameters
# declare -a beta_list=('1e-10' '1e-7' '1e-4' '1e-2' '1e0')
declare -a beta_list=('1e-4' '1e-1')

# declare -a l2_scale_list=('1e-10' '1e-5' '1e-2' '1e0')
declare -a l2_scale_list=('1e-5' '1e-3' '1e-2' '1e-1')

declare -a l1_scale_list=('0')

declare -a l1_scale_h1_list=('0' '1e-4' '1e-3' '1e-2' '1e-1')
# declare -a l1_scale_h1_list=('0' '1e-')

declare -a random_seed_list=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10')
# declare -a random_seed_list=('1' '2' '3')


# Running parameters
lr='1e-3'
test_size='0.1'
mb_size='2000'
iter='10000'
sample_size='50000'

for network_setup in "${network_setup_list[@]}"; do
  for rho in "${rho_list[@]}"; do
    adjdir="${logdir}/network_setup-${network_setup}/rho-${rho}"
    mkdir -p ${adjdir}
    python adjacency.py --logdir ${adjdir} \
                        --network_setup ${network_setup}
    /home/yongin/R-3.5.0/bin/Rscript adj_to_cov.R ${adjdir} ${rho}
    for h1_dim in "${h1_dim_list[@]}"; do
      for Z_dim in "${Z_dim_list[@]}"; do
        for beta in "${beta_list[@]}"; do
          for l2_scale in "${l2_scale_list[@]}"; do
            for l1_scale in "${l1_scale_list[@]}"; do
              for l1_scale_h1 in "${l1_scale_h1_list[@]}"; do
                for random_seed in "${random_seed_list[@]}"; do
                  dir="${adjdir}/Z_dim-${Z_dim}/h1_dim-${h1_dim}/beta-${beta}/l2-${l2_scale}/l1-${l1_scale}_h1-${l1_scale_h1}/random_seed-${random_seed}/"
                  mkdir -p ${dir}
                  echo ${dir}
                  ## run Model if it hasn't been run before
                  file="${dir}analysisZ3_2.png"
                  if [[ ! -e "${file}" ]]; then
                    echo "Run"
                    python run.py   --network_setup ${network_setup} \
                                    --Z_dim ${Z_dim} \
                                    --h1_dim ${h1_dim} \
                                    --beta ${beta} \
                                    --l1_scale ${l1_scale} \
                                    --l1_scale_h1 ${l1_scale_h1} \
                                    --l2_scale ${l2_scale} \
                                    --adjdir ${adjdir} \
                                    --logdir ${dir} \
                                    --lr ${lr} \
                                    --test_size ${test_size} \
                                    --mb_size ${mb_size} \
                                    --iter ${iter} \
                                    --random_seed ${random_seed} \
                                    --sample_size ${sample_size}
                  else
                    echo "Exists"
                  fi
                  sleep 0.5
                done
                python analyze_runs.py --logdir ${dir}
              done
            done
          done
        done
      done
    done
  done
done
