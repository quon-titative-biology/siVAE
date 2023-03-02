
# scalable, interpretable Variational Autoencoder (siVAE)



siVAE is an extension of traditional VAE that learns feature embeddings that guide the interpretation of the sample embeddings, in a manner analogous to factor loadings of factor analysis/PCA. siVAE is as powerful and nearly as fast to train as the standard VAE, but achieves full interpretability of the latent dimensions, as well as all hidden layers of the decoder. In addition, siVAE uses similarity between embeddings and gene regulatory networks to infer aspects of GRN.

This implementation of siVAE includes various analysis to visualize the interpretations of information learned by the modified VAE including new interpretability measure, feature awareness.


## Requirements
Operation systems: Linux
Programing language: Python 3.6


## Installation

siVAE requires installation of siVAE package as well as modified deepexplain from (https://github.com/marcoancona/DeepExplain), tensorflow-forward-ad (https://github.com/renmengye/tensorflow-forward-ad), and scvi-tools (https://github.com/YosefLab/scvi-tools)

Install siVAE by running the following command on the package file.

```
pip install siVAE
```

The package requires tensorflow >= 1.15 and tensorflow probability. In addition, installation of tensorflow-gpu is recommended for faster performance.

The installation typically takes under an hour.

## Running the model

Example of applying siVAE on subset of fetal liver dataset is shown in jupyter notebook.

[tutorial](tutorial/tutorial.md)

The tutorial will lead through how to visualize and interpret the cell/feature embeddings (similar to Fig.3a and b) from trained siVAE model. In addition, it will display table form of predicted gene degree centrality and neighborhood genes used to generate figures similar to Figure 5.
