from ann import layers
from ann.weightInitializers import Inintializers
from ann import models

from ann import lossFunctions
from ann import optimizers

from datasets import loader

import historyVisualizer


def trainModel(model, dataset, epoch: int, batch_size: int):
    test_x, test_y = dataset.loadTestDataset(batch_size=1, is_shuffle=False)

    total_train_losses = []
    total_test_losses = []
    for e in range(epoch):
        print(f'EPOCH ({e+1}/{EPOCH})')
        train_x, train_y = dataset.loadTrainDataset(batch_size=batch_size, is_shuffle=True)

        train_losses = model.train(x=train_x, y=train_y)
        test_losses = model.inference(x=test_x, y=test_y)
        print()

        test_loss = sum(test_losses) / len(test_losses)

        total_train_losses.extend(train_losses)
        total_test_losses.append(test_loss)
    return total_train_losses, total_test_losses


if __name__ == '__main__':
    EPOCH = 10
    BATCH_SIZE = 128 #64




    def create_BP_net():
        inputs = layers.InputLayer(shape=(784, ))
        out = layers.BPLayer(input_shape=784, units=1000, weight_init=Inintializers.Xavier)(inputs)
        out = layers.Sigmoid()(out)

        out = layers.BPLayer(input_shape=1000, units=10, weight_init=Inintializers.Xavier)(out)
        out = layers.Softmax()(out)

        model = models.Model(inputs=inputs, outputs=out)
        return model

    def create_FA_net():
        inputs = layers.InputLayer(shape=(784, ))
        out = layers.FALayer(input_shape=784, units=1000, weight_init=Inintializers.Xavier)(inputs)
        out = layers.Sigmoid()(out)

        out = layers.FALayer(input_shape=1000, units=10, weight_init=Inintializers.Xavier)(out)
        out = layers.Softmax()(out)

        model = models.Model(inputs=inputs, outputs=out)
        return model


    dataset = loader.Mnist(is_normalize=True, is_one_hot=True)
    test_x, test_y = dataset.loadTestDataset(batch_size=1, is_shuffle=False)

    optimizer = optimizers.SGD(learning_rate=0.001)
    # lossFunction = lossFunctions.SparseCrossEntropy(class_num=10)
    lossFunction = lossFunctions.SE()
    # lossFunction = lossFunctions.CrossEntropy()

    bp_model = create_BP_net()
    fa_model = create_FA_net()

    bp_model.compile(lossFunction=lossFunction, optimizer=optimizer)
    fa_model.compile(lossFunction=lossFunction, optimizer=optimizer)

    bp_train_his, bp_test_his = trainModel(model=bp_model, dataset=dataset, epoch=EPOCH, batch_size=BATCH_SIZE)
    fa_train_his, fa_test_his = trainModel(model=fa_model, dataset=dataset, epoch=EPOCH, batch_size=BATCH_SIZE)
    
    historyVisualizer.visualize(
        train_losses={'BP': bp_train_his, 'FA': fa_train_his},
        test_losses={'BP': bp_test_his, 'FA': fa_test_his},
        epoch=EPOCH
    )