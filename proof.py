from ann import layers
from ann.weightInitializers import Inintializers
from ann import models
from ann import lossFunctions
from ann import optimizers

if __name__ == '__main__':
    layer1 = layers.BPLayer(input_shape=1, units=1, weight_init=Inintializers.Xavier)
    layer2 = layers.Sigmoid()

    layer3 = layers.BPLayer(input_shape=1, units=2, weight_init=Inintializers.Xavier)
    layer4 = layers.Sigmoid()
    
    optimizer = optimizers.SGD(learning_rate=0.5)
    lossFunction = lossFunctions.MSE()

    x = [[0], [1]]
    y = [[1, 0], [0, 1]]

    x = [[0]]
    y = [[1, 0]]

    print('x:', x)
    print('y:', y, end='\n\n')

    print('layer1 W:', layer1.W)
    print('layer1 b:', layer1.b, end='\n\n')
    print('layer3 W:', layer3.W)
    print('layer3 b:', layer3.b)
    print('=' * 50)

    l1_o = layer1.forwardProp(x=x)
    l2_o = layer2.forwardProp(x=l1_o)
    l3_o = layer3.forwardProp(x=l2_o)
    l4_o = layer4.forwardProp(x=l3_o)
    loss = lossFunction.forwardProp(y_hat=l4_o, y=y)

    print('l1_o:', l1_o, end='\n\n')
    print('l2_o:', l2_o, end='\n\n')
    print('l3_o:', l3_o, end='\n\n')
    print('l4_o:', l4_o, end='\n\n')
    print('loss:', loss, end='\n\n')
    print('=' * 50)

    dLoss = lossFunction.backProp()
    dL4_o = layer4.backProp(dy=dLoss)
    dL3_o = layer3.backProp(dy=dL4_o)
    dL2_o = layer2.backProp(dy=dL3_o)
    dL1_o = layer1.backProp(dy=dL2_o)

    print('dLoss:', dLoss, end='\n\n')
    print('dL4_o:', dL4_o, end='\n\n')
    print('dL3_o:', dL3_o, end='\n\n')
    print('dL2_o:', dL2_o, end='\n\n')
    print('dL1_o:', dL1_o, end='\n\n')
    print('=' * 50)

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