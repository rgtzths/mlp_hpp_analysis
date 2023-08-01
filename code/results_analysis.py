import numpy as np
import pandas as pd 
from fanova import fANOVA
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
le = LabelEncoder()

datasets = {"All Datasets" : ["abalone", "bike_sharing", "compas", "covertype", "delays_zurich", "higgs"],
            "Regression" : ["abalone", "bike_sharing", "delays_zurich"],
            "Classification" : ["compas", "covertype", "higgs"]}

metrics = ["Performance", "Training time", "Inference time"]
average = []
for test in datasets:
    for dataset in datasets[test]:
        print(f"| | {dataset.capitalize()} | | |")
        print(f"| Hyperparameter | Performance | Training Time | Inference Time |")
        print("|---|---|---| ---- |")
        
        results = pd.read_csv(f'./results/final/{dataset}_results.csv', header=0)

        results["activation_functions"] = le.fit_transform([x.split(',')[1] for x in results["activation_functions"]])
        results["batch_size"] = le.fit_transform(results["batch_size"])
        results["loss"] = le.fit_transform(results["loss"])
        results["optimizer"] = le.fit_transform(results["optimizer"])
        results["learning_rate"] = le.fit_transform(results["learning_rate"])

        results["hidden_layer_dim"] = [len(x.split(",")) for x in results["hidden_layer_dims"]]
        results["hidden_layer_size"] = le.fit_transform([x.split(",")[1] for x in results["hidden_layer_dims"]])

        del results["hidden_layer_dims"]
        results = results[list(results.columns)[:5] + list(results.columns)[8:] + list(results.columns)[5:8]]
        X = results.values[:,:7]
        fanova_results = []
        for idx in range(3):
            fanova_results.append([])
            Y = results.values[:,7+idx]
            if Y[0] < 0:
                Y*= -1
            f = fANOVA(X,Y)
            if len(average) < idx+1:
                average.append([])
            for j in range(0,7):
                if len(average[idx]) < j+1:
                    average[idx].append(0)
                res = f.quantify_importance((j,))
                average[idx][j] += res[(j,)]['individual importance']
                fanova_results[-1].append(round(res[(j,)]['individual importance']*100,2))
        for j in range(7):
            row = f"| {results.columns[j]} | "
            for i in range(3):
                row += f"{fanova_results[i][j]} | "
            print(row)
        print("\n \n")
    print("\n \n")
    print(f"| | {test} | | |")
    print(f"| Hyperparameter | Performance | Training Time | Inference Time |")
    print("|---|---|---| ---- |")
    for j in range(7):
        row = f"| {results.columns[j]} | "
        for i in range(3):
            row += f"{round(average[i][j]/len(datasets[test])*100,2)} | "
        print(row)

    print("\n \n")