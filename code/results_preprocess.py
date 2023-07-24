
import pandas as pd


datasets = ["abalone", "bike_sharing", "compas", "covertype", "delays_zurich", "higgs"]

for dataset in datasets:
    results_file = f"results/raw/{dataset}_results.csv"

    df = pd.read_csv(results_file)

    del df['classifier']

    del df['callbacks']

    del df['epochs']

    del df['input_shape']

    del df['task_nodes']

    del df['Performance (Std)']

    del df['Training Time (Std)']

    del df['dataset']

    del df['dropouts']

    del df['task_activation']

    df['optimizer'] = df['optimizer'].str.lower()
    df = df.drop_duplicates(subset=['activation_functions', 'batch_size', 'hidden_layer_dims', 'loss', 'optimizer', 'learning_rate'])

    df.to_csv(f"results/final/{dataset}_results.csv", index=None)
