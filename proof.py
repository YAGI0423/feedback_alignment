from ann import layers
from ann.weightInitializers import Inintializers
from ann import models
from ann import lossFunctions
from ann import optimizers

if __name__ == '__main__':
    layer = layers.Softmax()
    loss_func = lossFunctions.CrossEntropy()

    import numpy as np

    h = np.array([
        [0.1, 0.5, 0.4],
        [0.7, 0.3, -0.2]
    ])

    y = np.array([
        [0., 0., 1.],
        [0., 1., 0.]
    ])


    y_hat = layer.forwardProp(x=h)
    loss = loss_func.forwardProp(y_hat=y_hat, y=y)
    
    print('pro_dLayer:', y_hat - y, end='\n\n')
    
    dLoss = loss_func.backProp()
    dLayer = layer.backProp(dy=dLoss)


    print('h:', h)
    print('h_hat:', y_hat)
    print('y:', y)
    print('\n\n')
    print('loss:', loss)
    print('\n\n')
    
    print('\n\n')
    print('dLoss:', dLoss)
    print('dLayer:', dLayer)


    exit()

    def get_model():
        inputs = layers.InputLayer(shape=(1, ))
        out = layers.BPLayer(input_shape=1, units=1, weight_init=Inintializers.Xavier)(inputs)
        out = layers.Sigmoid()(out)

        out = layers.BPLayer(input_shape=1, units=2, weight_init=Inintializers.Xavier)(out)
        out = layers.Softmax()(out)

        model = models.Model(inputs=inputs, outputs=out)
        return model

    model = get_model()
    model.optimizer = optimizers.SGD(learning_rate=0.1)

    lossFunction = lossFunctions.BinaryCrossEntropy()

    x = [[0], [1]]
    y = [[1, 0], [0, 1]]

    pre = model.predict(x=x)
    print(pre)
    
    exit()

    for _ in range(20000):
        y_hat = model.predict(x)
        loss = lossFunction.forwardProp(y_hat=y_hat, y=y)
        print(loss)

        dLoss = lossFunction.backProp()
        model.update_on_batch(dLoss)

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
        pass