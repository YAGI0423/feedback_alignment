from ann import layers
from ann.weightInitializers import Inintializers
from ann import models
from ann import lossFunctions
from ann import optimizers

from mnist_dataset import datasetLoader

from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == '__main__':
    def get_model():
        inputs = layers.InputLayer(shape=(784, ))
        out = layers.BPLayer(input_shape=784, units=1000, weight_init=Inintializers.Xavier)(inputs)
        out = layers.Sigmoid()(out)

        out = layers.BPLayer(input_shape=1000, units=10, weight_init=Inintializers.Xavier)(out)
        out = layers.Softmax()(out)

        model = models.Model(inputs=inputs, outputs=out)
        return model

    def train_model(x, y, model, loss_function):
        batch_size = len(x)
        dataset_iter = tqdm(zip(x, y), total=batch_size)

        losses = []
        for x, y in dataset_iter:
            y_hat = model.predict(x=x)
            loss = loss_function.forwardProp(y_hat=y_hat, y=y)

            dLoss = loss_function.backProp()
            model.update_on_batch(dLoss)

            dataset_iter.set_description(f'Loss: {loss:.5f}')
            losses.append(loss)
        return losses

    def inference_model(x, y, model, loss_function):
        batch_size = len(x)
        dataset_iter = tqdm(zip(x, y), total=batch_size)
        
        losses = []
        for x, y in dataset_iter:
            y_hat = model.predict(x=x)
            loss = loss_function.forwardProp(y_hat=y_hat, y=y)

            dataset_iter.set_description(f'Loss: {loss:.5f}')
            losses.append(loss)
        return losses
        

    dataset = datasetLoader.Loader(is_normalize=True)

    train_x, train_y = dataset.loadTrainDataset(batch_size=64, is_shuffle=True)
    test_x, test_y = dataset.loadTestDataset(batch_size=1, is_shuffle=False)
    
    model = get_model()

    model.optimizer = optimizers.SGD(learning_rate=0.001)
    lossFunction = lossFunctions.SparseCrossEntropy(class_num=10)

    EPOCH = 3
    BATCH_SIZE = 64

    total_train_losses = []
    test_loss_epoch = []
    for e in range(EPOCH):
        print(f'EPOCH ({e+1}/{EPOCH})')
        train_x, train_y = dataset.loadTrainDataset(batch_size=BATCH_SIZE, is_shuffle=True)

        train_losses = train_model(x=train_x, y=train_y, model=model, loss_function=lossFunction)
        test_losses = inference_model(x=test_x, y=test_y, model=model, loss_function=lossFunction)
        print()

        test_loss = sum(test_losses) / len(test_losses)

        total_train_losses.extend(train_losses)
        test_loss_epoch.append(test_loss)




    plt.figure(figsize=(9, 3))
    plt.subplot(1, 2, 1)
    plt.plot(total_train_losses)

    plt.subplot(1, 2, 2)
    plt.plot(test_loss_epoch)

    plt.show()
    