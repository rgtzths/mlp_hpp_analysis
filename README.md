# mlp_hpp_analysis
This repository is the code basis for the paper intitled "Exploring the Intricacies of Neural Network Optimization"

## Before using

Install `requirements.txt` by using the command `pip install -r requirements.txt`

## To use this module

1. Write various `.json` files with the experiments you want to perform.

2. Run the experiments using the comand `python code/run.py --hyper path_to_the_folder`

## Execute the experiments performed in the paper

In the `hyperparameters` folder there is one folder for each of the tested datasets.

If the user desires to run every experiment at the same time use the `all_runs` folder.
Otherwise it can run the experiments by folder individually achieving the same results as the ones presented in the paper.

Keep in mind the experiments with the `binary_crossentropy` and `sparse_categorical_crossentropy` are kept in a seperate folder as they require Y array to be created differently.
You can run them seperatly and then join the csv results.

With the experiments performed the results should be presented in `results/raw` folder.

To preprocess them run `python code/results_preprocess.py` which should create the `results/final` folder with the preprocessed results.

After that to obtain the importance of the hyperparameters run `python code/results_analysis.py` which should present the importance by dataset and the average of the six datasets.

## Results

Here we present the results that are available in the paper and an additional analysis of the obtained results.

If there is any analysis missing that the reader might desire to perform, the complete data obtained from the runs is available in the `results` folder, or the reader might run the experiments him self.

### Hyperparameter importance

These are the results of the fANOVA analysis.

#### General Importance

| | All Datasets | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| activation_functions | 18.42 | 3.2 | 6.99 | 
| batch_size | 0.95 | 55.94 | 37.67 | 
| loss | 12.23 | 0.33 | 2.1 | 
| optimizer | 14.88 | 5.17 | 2.16 | 
| learning_rate | 17.65 | 3.38 | 1.34 | 
| hidden_layer_dim | 3.94 | 3.85 | 16.62 | 
| hidden_layer_size | 3.94 | 3.61 | 6.29 | 

#### Importance by dataset type

| |Classification| | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| activation_functions | 78.09 | 12.18 | 32.23 | 
| batch_size | 7.69 | 234.05 | 150.0 | 
| loss | 53.14 | 0.78 | 7.97 | 
| optimizer | 54.26 | 22.45 | 8.98 | 
| learning_rate | 74.78 | 12.82 | 5.95 | 
| hidden_layer_dim | 16.12 | 10.57 | 64.82 | 
| hidden_layer_size | 14.23 | 13.9 | 25.3 |

| | Regression | | |
|---|---|---| ---- |
| Hyperparameter | Performance | Training Time | Inference Time |
| activation_functions | 60.5 | 9.27 | 29.96 | 
| batch_size | 6.38 | 176.75 | 112.57 | 
| loss | 43.97 | 0.77 | 4.21 | 
| optimizer | 37.15 | 18.67 | 4.45 | 
| learning_rate | 53.38 | 8.14 | 5.94 | 
| hidden_layer_dim | 9.99 | 9.9 | 45.45 | 
| hidden_layer_size | 11.19 | 8.7 | 16.76 | 

#### Importance per dataset

| | |Abalone| |
| Hyperparameter | Performance | Training Time | Inference Time |
|---|---|---| ---- |
| activation_functions | 14.77 | 1.39 | 4.39 | 
| batch_size | 0.55 | 56.72 | 21.61 | 
| loss | 0.0 | 1.62 | 0.0 | 
| optimizer | 2.96 | 7.99 | 3.5 | 
| learning_rate | 30.02 | 6.9 | 0.07 | 
| hidden_layer_dim | 7.16 | 0.12 | 15.69 | 
| hidden_layer_size | 11.55 | 4.35 | 11.04 | 

 

| | |Bike_sharing| |
| Hyperparameter | Performance | Training Time | Inference Time |
|---|---|---| ---- |
| activation_functions | 51.26 | 0.59 | 24.54 | 
| batch_size | 0.74 | 72.21 | 29.71 | 
| loss | 0.06 | 0.0 | 0.0 | 
| optimizer | 17.86 | 6.28 | 0.02 | 
| learning_rate | 11.6 | 5.17 | 7.14 | 
| hidden_layer_dim | 0.0 | 1.98 | 14.41 | 
| hidden_layer_size | 2.62 | 1.16 | 0.82 | 

 

| | |Compas| |
| Hyperparameter | Performance | Training Time | Inference Time |
|---|---|---| ---- |
| activation_functions | 3.4 | 0.4 | 0.08 | 
| batch_size | 1.16 | 43.0 | 6.23 | 
| loss | 33.98 | 0.19 | 0.0 | 
| optimizer | 21.68 | 4.02 | 4.16 | 
| learning_rate | 9.59 | 6.06 | 0.02 | 
| hidden_layer_dim | 0.76 | 2.92 | 49.31 | 
| hidden_layer_size | 3.61 | 7.49 | 20.06 | 

 

| | |Covertype| |
| Hyperparameter | Performance | Training Time | Inference Time |
|---|---|---| ---- |
| activation_functions | 29.22 | 12.77 | 4.01 | 
| batch_size | 0.77 | 56.92 | 41.6 | 
| loss | 0.06 | 0.0 | 10.34 | 
| optimizer | 8.29 | 1.65 | 4.67 | 
| learning_rate | 23.64 | 0.32 | 0.17 | 
| hidden_layer_dim | 13.27 | 0.2 | 3.32 | 
| hidden_layer_size | 1.84 | 4.79 | 0.62 | 

 

| | |Delays_zurich| |
| Hyperparameter | Performance | Training Time | Inference Time |
|---|---|---| ---- |
| activation_functions | 0.37 | 3.57 | 5.2 | 
| batch_size | 0.0 | 58.2 | 57.82 | 
| loss | 39.27 | 0.0 | 0.01 | 
| optimizer | 14.39 | 2.42 | 0.0 | 
| learning_rate | 0.18 | 0.58 | 0.5 | 
| hidden_layer_dim | 2.37 | 10.18 | 12.22 | 
| hidden_layer_size | 3.81 | 0.48 | 3.92 | 

 

| | |Higgs| |
| Hyperparameter | Performance | Training Time | Inference Time |
|---|---|---| ---- |
| activation_functions | 11.51 | 0.49 | 3.73 | 
| batch_size | 2.46 | 48.6 | 69.07 | 
| loss | 0.01 | 0.14 | 2.25 | 
| optimizer | 24.08 | 8.67 | 0.63 | 
| learning_rate | 30.84 | 1.22 | 0.15 | 
| hidden_layer_dim | 0.09 | 7.68 | 4.75 | 
| hidden_layer_size | 0.18 | 3.39 | 1.25 |

### Performance metrics

#### Best performing hyperparameter combination per dataset

#### Baseline vs Best vs Worst performance comparison



## Authors

* **Rafael Teixeira** - [rgtzths](https://github.com/rgtzths)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details