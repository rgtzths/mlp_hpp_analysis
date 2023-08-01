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

#### Importance by dataset type

#### Importance per dataset

### Performance metrics

#### Best performing hyperparameter combination per dataset

#### Baseline vs Best vs Worst performance comparison



## Authors

* **Rafael Teixeira** - [rgtzths](https://github.com/rgtzths)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details