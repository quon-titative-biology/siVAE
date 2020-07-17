# scalable, interpretable Variational Autoencoder (siVAE)

siVAE is an extension of traditional VAE that learns feature embeddings that guide the interpretation of the sample embeddings, in a manner analogous to factor loadings of factor analysis/PCA. siVAE is as powerful and nearly as fast to train as the standard VAE, but achieves full interpretability of the latent dimensions, as well as all hidden layers of the decoder.

This implementation of siVAE includes various analysis to visualize the interpretations of information learned by the modified VAE including new interpretability measure, feature awareness.


## Installation

siVAE requires installation of siVAE package as well as modified deepexplain from (https://github.com/marcoancona/DeepExplain).

Install siVAE by running the following command from `supplementary/code` directory.


```
pip install -e siVAE
```

The package requires tensorflow >= 1.15 and tensorflow probability. In addition, installation of tensorflow-gpu is recommended for faster performance.

In addition, install modified version of deepexplain originally retrieved from https://github.com/marcoancona/DeepExplain

```
pip install -e deepexplain
```



## Example

An example case of running siVAE  on MNIST is shown in ```supplementary/code/siVAE/run_MNIST.py```.


### Running the model

The entire code can be run through the following command:
```
python3 run_MNIST.py
```

Here is the step by step guide of the example code.

The first section specifies parameters and run options for the model. Setting `do_FA = True` will run traditional feature attribution methods to be compared against siVAE. This significantly increase the run time.

After setting the parameters, we run the following function to load preprocessed MNSIT as well as additional information such as dimension of images, arguments for plotting, and data handler that handles data.
```
kwargs_FI, sample_set, datah_feature, datah_sample,palette,plot_args_dict_sf,ImageDims,data_name = load_MNIST.prepare_data(architecture=architecture,LE_dim=LE_dim,beta=beta)
```

Then we run the model which returns `result_dict`, a dictionary of results with evaluations included as losses.

```
result_dict = run_VAE(logdir, graph_args, LE_method, datah_sample,
                      zv_recon_scale=zv_recon_scale, datah_feature=datah_feature,
                      do_pretrain=True, output=output, method_DE=method_DE,
                      sample_set=sample_set, VAE_FI = do_FA,
                      add_classifier = False, labels = plot_args_dict_sf['sample']['labels'],
                      kwargs_FI = kwargs_FI)
```
### Analysis and Visualization

The results are analyzed and visualized with following set up functions.

The feature-wise and sample-wise latent embeddings can be plotted in `ScatterLE.pdf`.
```
analysis.plot_latent_embeddings(values_dict,
                                plot_args_dict_sf,
                                palette,logdir=logdir)
```

The loadings can be visualized in `Loadings-siVAE.pdf` with the following function.
```
analysis.plot_siVAE_loadings(values_dict, logdir, ImageDims)
```

The inferred loadings can also be visualized in `Loadings-encoder.pdf` or `Loadings-encoder.pdf` with the following function. This requires `do_FA` to be true.
```
analysis.plot_FA_loadings(values_dict, logdir, ImageDims)
```

The recoded embeddings that capture the variance within a subset of data consisting of same class is plotted in `Recoded_embeddings-l-{level}.pdf`, where level is the index of the hidden layer in the decoder starting with the latent dimension at 1. It also plots the variance along the recoded principal component in `Recoded_embeddings_label-{class}.pdf`.

```
analysis.recode_embeddings(values_dict,
                           plot_args_dict_sf,
                           ImageDims, n_pc=3,
                           logdir=logdir_ks)
```

Finally we plot the feature awareness that measures captures which features each layer of the siVAE model focuses on reconstructing well for each input sample, which is shown in `FeatureAwareness.pdf`

```
analysis.plot_feature_awareness(result_dict,
                                scaler,ImageDims,
                                plot_args_dict_sf)
```

## Detailed
# siVAE()
