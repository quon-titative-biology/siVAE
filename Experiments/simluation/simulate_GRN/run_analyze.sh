#!/bin/bash

## Directories
logdir="/home/yongin/Gen/181130_IntMethod_test/1212/out/exp10"

## Parameter Lists for Looping
# Network Setup
# declare -a network_setup_list=('0_5_5_5' '5_5_5' '0_5_5_15')
# declare -a network_setup_list=('0_5_5_15' '0_5_5_5' '5_5_5')
declare -a network_setup_list=('0_5_5_15' '5_5_15' '0_5_5_5')

# declare -a overlap_list=('1' '0')
declare -a overlap_list=('1' '0')

# declare -a rho_list=('0.4' '0.6' '0.8')
declare -a rho_list=('0.6' '0.8')

# declare -a Z_dim_list=("3" "2")
declare -a Z_dim_list=("2" "3" "5")

declare -a h1_dim_list=("10")

# Hyperparameters
# declare -a beta_list=('1e-10' '1e-7' '1e-4' '1e-2' '1e0')
declare -a beta_list=('1e-4' '1e-1')

# declare -a l2_scale_list=('1e-10' '1e-5' '1e-2' '1e0')
declare -a l2_scale_list=('1e-10' '1e-5')

declare -a l1_scale_list=('0')

# declare -a l1_scale_h1_list=('0' '1e-4' '1e-3' '1e-2' '1e-1')
declare -a l1_scale_h1_list=('0' '1e-2')
# declare -a l1_scale_h1_list=('0')

# declare -a random_seed_list=('1' '2' '3' '4' '5' '101' '102' '103' '104' '105')
# declare -a random_seed_list=('1' '2' '3')
random_seed='1'

# Running parameters
lr='1e-3'
test_size='0.1'
mb_size='2000'
iter='10000'
sample_size='50000'

for network_setup in "${network_setup_list[@]}"; do
  for overlap in "${overlap_list}"; do
    for rho in "${rho_list[@]}"; do
      for h1_dim in "${h1_dim_list[@]}"; do
        for Z_dim in "${Z_dim_list[@]}"; do
          for beta in "${beta_list[@]}"; do
            for l2_scale in "${l2_scale_list[@]}"; do
              for l1_scale in "${l1_scale_list[@]}"; do
                for l1_scale_h1 in "${l1_scale_h1_list[@]}"; do
                  adjdir="${logdir}/network_setup-${network_setup}/overlap-${overlap}/rho-${rho}"
                  dir="${adjdir}/Z_dim-${Z_dim}/h1_dim-${h1_dim}/beta-${beta}/l2-${l2_scale}/l1-${l1_scale}_h1-${l1_scale_h1}/random_seed-${random_seed}/"
                  python analyze_runs.py --logdir ${dir}
                done
              done
            done
          done
        done
      done
    done
  done
done
