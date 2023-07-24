import numpy as np
import pandas as pd 
from fanova import fANOVA
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
le = LabelEncoder()

datasets = ["abalone", "bike_sharing", "compas", "covertype", "delays_zurich", "higgs"]
#datasets = ["abalone", "bike_sharing", "delays_zurich"]
#datasets = ["compas", "covertype", "higgs"]

metrics = ["Performance", "Training time", "Inference time"]
average = []
for dataset in datasets:
    print("\n-----------", dataset.capitalize(), "-------------\n")
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
    for idx in range(3):
        print("\n-----", metrics[idx], "--------\n")
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
            print(results.columns[j], round(res[(j,)]['individual importance']*100,2))

print("\n---------------Average importance------------- \n")
for idx, importance in enumerate(average):
    print("\n-----", metrics[idx], "--------\n")

    for idx2, metric in enumerate(importance):
        print(results.columns[idx2], round(metric/len(datasets)*100,2))