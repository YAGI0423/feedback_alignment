from abc import abstractmethod, ABCMeta
import ann.layers as layers

import numpy as np

class Model:
    def __init__(self, inputs, outputs):
        self.input_layer = inputs
        self.output_layer = outputs

    def predict(self, x):
        layers = self.input_layer
        flow_data = x
        while layers is not None:
            flow_data = layers.forwardProp(x=flow_data)
            layers = layers.childLayer
        return flow_data



if __name__ == '__main__':
    inputs = layers.InputLayer(shape=(2, ))
    out = layers.BPLayer(input_shape=2, units=1)(inputs)
    print(out.W)
    print(out.b)
    out = layers.Sigmoid()(out)

    out = layers.BPLayer(input_shape=1, units=1)(out)
    print(out.W)
    print(out.b)
    out = layers.Sigmoid()(out)

    model = Model(inputs=inputs, outputs=out)
    out = model.predict(x=[[0, 0], [0, 1]])

    print('out: ', out)
    exit()
    faLayer = layers.BPLayer(input_shape=2, units=1)
    sigmoid = layers.Sigmoid()
    loss_func = layers.MSE()

    print('W: ', faLayer.W, end='\n\n')
    print('b: ', faLayer.b, end='\n\n')

    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [1]]

    print('x: ', x, end='\n\n')
    print('y: ', y, end='\n\n')

    h = faLayer.forwardProp(x=x)
    print('h: ', h, end='\n\n')

    o = sigmoid.forwardProp(x=h)
    print('o: ', o, end='\n\n')

    loss = loss_func.forwardProp(y_hat=o, y=y)
    print('loss: ', loss, end='\n\n')

    print('=' * 50)

    do = loss_func.backProb()
    print('do: ', do, end='\n\n')

    dh = sigmoid.backProb(dy=do)
    print('dh: ', dh, end='\n\n')

    dx, dW, db = faLayer.backProb(dy=dh)
    print('dx: ', dx, end='\n\n')
    print('dW: ', dW, end='\n\n')
    print('db: ', db, end='\n\n')

    print('=' * 50)
    LEARNING_RATE = 0.5

    faLayer.W -= LEARNING_RATE * dW
    faLayer.b -= LEARNING_RATE * db

    print('x: ', x, end='\n\n')
    print('y: ', y, end='\n\n')

    h = faLayer.forwardProp(x=x)
    print('h: ', h, end='\n\n')

    o = sigmoid.forwardProp(x=h)
    print('o: ', o, end='\n\n')

    loss = loss_func.forwardProp(y_hat=o, y=y)
    print('loss: ', loss, end='\n\n')
