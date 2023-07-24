import pandas as pd

from datasets import load_dataset

'''
    Loads a csv file in the format feature1, feature2, ..., label.
    Divides it into 80% for training and 20% for testing.
    Returns ((x_train, y_train), (x_test, y_test))
'''
def load_dataset_from_file(file_location): 

    if file_location.split(".")[-1] != "csv":
        raise ValueError("Dataset should be a csv file.")

    data = pd.read_csv(file_location, index_col=False)

    train_length = int(data.shape[0]*0.8)

    x_train = data.iloc[:train_length,:data.shape[1]-1].values
    y_train = data.iloc[:train_length, data.shape[1]-1].values

    x_test = data.iloc[train_length: , :data.shape[1]-1].values
    y_test = data.iloc[train_length: , data.shape[1]-1].values

    return (x_train, y_train), (x_test, y_test)

def load_covertype(): #Tabular classification categoric_numeric (high number of examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="clf_cat/covertype.csv", split="train")
    dataset = dataset.to_pandas().values
    train_length = int(dataset.shape[0]*0.8)

    x_train = dataset[:train_length,:-1]
    y_train = dataset[:train_length,-1] -1

    x_test = dataset[train_length:,:-1]
    y_test = dataset[train_length:,-1] -1

    return (x_train, y_train), (x_test, y_test)

def load_higgs(): #Tabular classification numeric (high number of examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="clf_num/Higgs.csv", split="train")
    dataset = dataset.to_pandas().values
    train_length = int(dataset.shape[0]*0.8)

    x_train = dataset[:train_length,:-1]
    y_train = dataset[:train_length,-1]

    x_test = dataset[train_length:,:-1]
    y_test = dataset[train_length:,-1]

    return (x_train, y_train), (x_test, y_test)

def load_compas(): #Tabular classification categoric (low examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="clf_cat/compas-two-years.csv", split="train")
    dataset = dataset.to_pandas().values
    train_length = int(dataset.shape[0]*0.8)

    x_train = dataset[:train_length,:-1]
    y_train = dataset[:train_length,-1]

    x_test = dataset[train_length:,:-1]
    y_test = dataset[train_length:,-1]

    return (x_train, y_train), (x_test, y_test) 

def load_delays_zurich(): #Tabular regression numeric
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="reg_num/delays_zurich_transport.csv", split="train")
    dataset = dataset.to_pandas().values
    train_length = int(dataset.shape[0]*0.8)

    x_train = dataset[:train_length,:-1]
    y_train = dataset[:train_length,-1]

    x_test = dataset[train_length:,:-1]
    y_test = dataset[train_length:,-1]

    return (x_train, y_train), (x_test, y_test) 

def load_abalone(): #Tabular regression mixture (low examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="reg_cat/abalone.csv", split="train")
    dataset = dataset.to_pandas().values
    train_length = int(dataset.shape[0]*0.8)

    x_train = dataset[:train_length,:-1]
    y_train = dataset[:train_length,-1]

    x_test = dataset[train_length:,:-1]
    y_test = dataset[train_length:,-1]

    return (x_train, y_train), (x_test, y_test) 


def load_bike_sharing(): #Tabular regression mixture (numerous examples, more cat then reg)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="reg_cat/Bike_Sharing_Demand.csv", split="train")
    dataset = dataset.to_pandas().values
    train_length = int(dataset.shape[0]*0.8)

    x_train = dataset[:train_length,:-1]
    y_train = dataset[:train_length,-1]

    x_test = dataset[train_length:,:-1]
    y_test = dataset[train_length:,-1]

    return (x_train, y_train), (x_test, y_test) 
