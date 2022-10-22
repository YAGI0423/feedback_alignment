from ann import layers
from ann.weightInitializers import Inintializers
from ann import models
from ann import lossFunctions
from ann import optimizers

if __name__ == '__main__':
    def get_model():
        inputs = layers.InputLayer(shape=(2, ))
        out = layers.BPLayer(input_shape=2, units=2, weight_init=Inintializers.xavier)(inputs)
        out = layers.Sigmoid()(out)

        out = layers.BPLayer(input_shape=2, units=1, weight_init=Inintializers.xavier)(out)
        out = layers.Sigmoid()(out)

        model = models.Model(inputs=inputs, outputs=out)
        return model

    model = get_model() 
    model.optimizer = optimizers.SGD(learning_rate=0.1)

    lossFunction = lossFunctions.BinaryCrossEntropy()

    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]

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