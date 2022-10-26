from ann import layers
from ann.weightInitializers import Inintializers
from ann import models
from ann import lossFunctions
from ann import optimizers

from mnist_dataset import datasetLoader

from tqdm import tqdm
import matplotlib.pyplot as plt


from abc import abstractmethod, ABCMeta
class ModelFrame(metaclass=ABCMeta):
    @abstractmethod
    def create_model(self):
        pass

    def train(self, x, y):
        batch_size = len(x)
        dataset_iter = tqdm(zip(x, y), total=batch_size)

        losses = []
        for x, y in dataset_iter:
            y_hat = self.model.predict(x=x)
            loss = self.lossF.forwardProp(y_hat=y_hat, y=y)

            dLoss = self.lossF.backProp()
            self.model.update_on_batch(dLoss)

            dataset_iter.set_description(f'Loss: {loss:.5f}')
            losses.append(loss)
        return losses

    def inference(self, x, y):
        batch_size = len(x)
        dataset_iter = tqdm(zip(x, y), total=batch_size)
        
        losses = []
        for x, y in dataset_iter:
            y_hat = self.model.predict(x=x)
            loss = self.lossF.forwardProp(y_hat=y_hat, y=y)

            dataset_iter.set_description(f'Loss: {loss:.5f}')
            losses.append(loss)
        return losses

class BPmodel(ModelFrame):
    def __init__(self, optimizer, lossFunction):
        self.model = self.create_model()
        self.model.optimizer = optimizer
        self.lossF = lossFunction

    def create_model(self):
        inputs = layers.InputLayer(shape=(784, ))
        out = layers.BPLayer(input_shape=784, units=1000, weight_init=Inintializers.Xavier)(inputs)
        out = layers.Sigmoid()(out)

        out = layers.BPLayer(input_shape=1000, units=10, weight_init=Inintializers.Xavier)(out)
        out = layers.Softmax()(out)

        model = models.Model(inputs=inputs, outputs=out)
        return model


if __name__ == '__main__':
    dataset = datasetLoader.Loader(is_normalize=True)
    test_x, test_y = dataset.loadTestDataset(batch_size=1, is_shuffle=False)

    optimizer = optimizers.SGD(learning_rate=0.001)
    lossFunction = lossFunctions.SparseCrossEntropy(class_num=10)

    model = BPmodel(optimizer=optimizer, lossFunction=lossFunction)

    EPOCH = 10
    BATCH_SIZE = 64

    total_train_losses = []
    test_loss_epoch = []
    for e in range(EPOCH):
        print(f'EPOCH ({e+1}/{EPOCH})')
        train_x, train_y = dataset.loadTrainDataset(batch_size=BATCH_SIZE, is_shuffle=True)

        train_losses = model.train(x=train_x, y=train_y)
        test_losses = model.inference(x=test_x, y=test_y)
        print()

        test_loss = sum(test_losses) / len(test_losses)

        total_train_losses.extend(train_losses)
        test_loss_epoch.append(test_loss)


    def pltDefaultSetting(plt, title: str, ticks, labels):
        title_args ={'fontsize': 13, 'fontweight': 'bold'}

        plt.title(title, fontdict=title_args, loc='left', pad=10)

        plt.xticks(ticks=ticks, labels=labels)
        plt.tick_params(axis='x', direction='in')
        plt.tick_params(axis='y', direction='in')

    

    plt.figure(figsize=(7, 9))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.925, wspace=0.1, hspace=0.3)

    plt.subplot(2, 1, 1)
    plt.plot(total_train_losses, color='k', label='BP')

    size_per_epoch = int(len(total_train_losses) / EPOCH) - 1
    
    ticks = tuple(e * size_per_epoch for e in range(0, EPOCH+1, 5))
    labels = tuple(range(0, EPOCH+1, 5))

    pltDefaultSetting(plt, title='<Loss on Train Set>', ticks=ticks, labels=labels)


    plt.subplot(2, 1, 2)
    plt.plot(test_loss_epoch, color='k', label='BP')

    ticks = list(range(-1, EPOCH+1, 5))
    ticks[0] = 0
    labels = list(range(0, EPOCH+1, 5))
    labels[0] = 1

    pltDefaultSetting(plt, title='<Loss on Test Set>', ticks=ticks, labels=labels)

    plt.show()
    