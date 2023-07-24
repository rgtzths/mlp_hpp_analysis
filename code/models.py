from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow.keras.layers import (GRU, LSTM, BatchNormalization, Conv1D,
                                     Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling1D, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import (SGD, Adadelta, Adagrad, Adam, Adamax,
                                         Ftrl, Nadam, RMSprop)


def get_optimizer(optimizer, learning_rate=None):

    if optimizer.lower() == "adam":
        return Adam(learning_rate=learning_rate) if learning_rate else Adam()
    elif optimizer.lower() == "rmsprop":
        return RMSprop(learning_rate=learning_rate) if learning_rate else RMSprop()
    elif optimizer.lower() == "sgd":
        return SGD(learning_rate=learning_rate) if learning_rate else SGD(learning_rate=0.00001)
    elif optimizer.lower() == "adadelta":
        return Adadelta(learning_rate=learning_rate) if learning_rate else Adadelta()
    elif optimizer.lower() == "adagrad":
        return Adagrad(learning_rate=learning_rate) if learning_rate else Adagrad()
    elif optimizer.lower() == "adamax":
        return Adamax(learning_rate=learning_rate) if learning_rate else Adamax()
    elif optimizer.lower() == "nadam":
        return Nadam(learning_rate=learning_rate) if learning_rate else Nadam()
    elif optimizer.lower() == "ftrl":
        return Ftrl(learning_rate=learning_rate) if learning_rate else Ftrl()
    else:
        raise ValueError(f'The optimizer {optimizer} is not supported!')
    

def create_dense_model(classification):

    def model(input_shape, hidden_layer_dims, activation_functions, dropouts, task_activation, task_nodes):
        model = Sequential()

        model.add(Dense(units=hidden_layer_dims[0], input_shape=input_shape, activation=activation_functions[0]))
        if dropouts[0] > 0:
            model.add(Dropout(dropouts[0]))
        for i in range(1, len(hidden_layer_dims)):
            model.add(Dense(units=hidden_layer_dims[i], activation=activation_functions[i]))
            if dropouts[i] > 0:
                model.add(Dropout(dropouts[i]))
        
        model.add(Dense(units=task_nodes,activation=task_activation))
        
        return model

    if classification:
        return KerasClassifier(model=model, 
                            verbose=0, 
                            input_shape=(768,),
                            hidden_layer_dims=[100],
                            activation_functions = ['relu'],
                            dropouts = [0],
                            task_activation = 'softmax',
                            task_nodes = 1,
                            optimizer='adam',
                            loss='categorical_crossentropy',
                            )

    return KerasRegressor(model=model, 
                            verbose=0, 
                            input_shape=(768,),
                            hidden_layer_dims=[100],
                            activation_functions = ['relu'],
                            dropouts = [0],
                            task_activation = 'linear',
                            task_nodes = 1,
                            optimizer='adam',
                            loss='mean_squared_error',
                            )