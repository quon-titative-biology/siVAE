#!/bin/bash -l

## =============================================================================
## 1. Run siVAE on MNIST with 2 digits for visualization
## =============================================================================

## siVAE models
python run_MNIST_subset.py --method "siVAE" \
                           --do_FA "True" \
                           --logdir 'out/siVAE'


## Run gene relevance
Rscript gene_relevance.R

## Plot
python GeneRelevanceVisualization.py

## =============================================================================
## 2. Run siVAE on 3 imaging datasets with 3 settings
## =============================================================================

##
python run_MNIST.py --method "siVAE" \
                    --do_FA "False" \
                    --logdir 'out/all/MNIST/siVAE' \
                    --dataset "MNIST"

python run_MNIST.py --method "siVAE-0" \
                    --do_FA "False" \
                    --logdir 'out/all/MNIST/siVAE-0' \
                    --dataset "MNIST"

## Fashion-MNIST
python run_MNIST.py --method "siVAE" \
                    --do_FA "False" \
                    --logdir 'out/all/FMNIST/siVAE' \
                    --dataset "FMNIST"

python run_MNIST.py --method "siVAE-0" \
                    --do_FA "False" \
                    --logdir 'out/all/FMNIST/siVAE-0' \
                    --dataset "FMNIST"

## CIFAR-10
python run_MNIST.py --method "siVAE" \
                    --do_FA "False" \
                    --logdir 'out/all/CIFAR10/siVAE-0' \
                    --dataset "CIFAR10"

python run_MNIST.py --method "siVAE-0" \
                    --do_FA "False" \
                    --logdir 'out/all/CIFAR10/siVAE-0' \
                    --dataset "CIFAR10"
