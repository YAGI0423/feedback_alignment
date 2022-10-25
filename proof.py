from ann import layers
from ann.weightInitializers import Inintializers
from ann import models
from ann import lossFunctions
from ann import optimizers

from mnist_dataset import datasetLoader

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == '__main__':
    def get_model():
        inputs = layers.InputLayer(shape=(784, ))
        out = layers.BPLayer(input_shape=784, units=1000, weight_init=Inintializers.He)(inputs)
        out = layers.ReLU()(out)

        out = layers.BPLayer(input_shape=1000, units=10, weight_init=Inintializers.Xavier)(out)
        out = layers.Softmax()(out)

        model = models.Model(inputs=inputs, outputs=out)
        return model

    def train_model(x, y, model, loss_function,shuffle: bool=False):
        batch_size = len(x)
        dataset_iter = tqdm(zip(x, y), total=batch_size)

        losses = []
        for x, y in dataset_iter:
            y_hat = model.predict(x=x)
            loss = loss_function.forwardProp(y_hat=y_hat, y=y)

            dLoss = lossFunction.backProp()
            model.update_on_batch(dLoss)

            dataset_iter.set_description(f'Loss: {loss:.5f}')
            losses.append(loss)
        return losses
        

    datset = datasetLoader.Loader(is_normalize=True)

    train_x, train_y = datset.loadTrainDataset(batch_size=32, is_shuffle=True)
    test_x, test_y = datset.loadTestDataset(batch_size=8, is_shuffle=False)
    
    model = get_model()

    model.optimizer = optimizers.SGD(learning_rate=0.001)
    lossFunction = lossFunctions.SparseCrossEntropy(class_num=10)

    losses = train_model(x=train_x, y=train_y, model=model, loss_function=lossFunction)

        
    
    

    plt.plot(losses)
    plt.show()
    exit()

    
    

    def get_model():
        inputs = layers.InputLayer(shape=(2, ))
        out = layers.BPLayer(input_shape=2, units=1, weight_init=Inintializers.Xavier)(inputs)
        out = layers.Sigmoid()(out)

        out = layers.BPLayer(input_shape=1, units=4, weight_init=Inintializers.Xavier)(out)
        out = layers.Softmax()(out)

        model = models.Model(inputs=inputs, outputs=out)
        return model

    model = get_model()
    model.optimizer = optimizers.SGD(learning_rate=0.1)

    lossFunction = lossFunctions.CrossEntropy()

    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    for _ in range(30000):
        y_hat = model.predict(x)
        loss = lossFunction.forwardProp(y_hat=y_hat, y=y)
        print(loss)

        dLoss = lossFunction.backProp()
        model.update_on_batch(dLoss)

    print(model.predict([[1, 1]]).round(2))
    # print(model.predict([[1, 1]]))
    
    # lay = model.input_layer
    # while lay is not None:
    #     try:
    #         print('name:', lay.__class__.__name__)
    #         print('W:', lay.W)
    #         print('b:', lay.b, end='\n\n')
    #     except:
    #         pass
    #     lay = lay.childLayer
    #     pass