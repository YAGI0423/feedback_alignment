from ann import layers
from ann.weightInitializers import Inintializers
from ann import models
from ann import lossFunctions
from ann import optimizers

if __name__ == '__main__':
    import numpy as np
    import pickle

    with open('./mnist_dataset/mnist_dataset.pk', 'rb') as fr:
        mnist_dataset = pickle.load(fr)

    x_train, y_train = mnist_dataset['x_train'], mnist_dataset['y_train']
    x_test, y_test = mnist_dataset['x_test'], mnist_dataset['y_test']

    def split_dataset(x, batch_size):
        dataset_size = x.shape[0]
        split_size = int(dataset_size / batch_size)
        return np.array_split(x, split_size)

    
    BATCH_SIZE = 32

    x_train, y_train = split_dataset(x_train, BATCH_SIZE), split_dataset(y_train, BATCH_SIZE)
    x_test, y_test = split_dataset(x_test, BATCH_SIZE), split_dataset(y_test, BATCH_SIZE)
    
    print(x_test[-1].shape)
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