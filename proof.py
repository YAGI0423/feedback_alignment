from ann import layers
from ann.weightInitializers import Inintializers
from ann import models
from ann import lossFunctions
from ann import optimizers

import numpy as np

if __name__ == '__main__':
    def get_model():
        inputs = layers.InputLayer(shape=(2, ))
        out = layers.BPLayer(input_shape=2, units=2, weight_init=Inintializers.He)(inputs)
        out = layers.ReLU()(out)

        out = layers.BPLayer(input_shape=2, units=1, weight_init=Inintializers.xavier)(out)
        out = layers.Sigmoid()(out)

        model = models.Model(inputs=inputs, outputs=out)
        return model

    model = get_model() 
    model.optimizer = optimizers.SGD(learning_rate=0.1)

    lossFunction = lossFunctions.MSE()

    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]

    for _ in range(30000):
        y_hat = model.predict(x)
        loss = lossFunction.forwardProp(y_hat=y_hat, y=y)
        print(loss)

        dLoss = lossFunction.backProb()
        model.update_on_batch(x, dLoss)

    print(model.predict([[0, 1]]))
    print(model.predict([[1, 1]]))
    
    lay = model.input_layer
    while lay is not None:
        try:
            print('name:', lay.__class__.__name__)
            print('W:', lay.W)
            print('b:', lay.b, end='\n\n')
        except:
            pass
        lay = lay.childLayer