from ann import lossFunctions
from ann import optimizers
from ann import validateModels

from mnist_dataset import datasetLoader

import historyVisualizer

if __name__ == '__main__':
    dataset = datasetLoader.Loader(is_normalize=True)
    test_x, test_y = dataset.loadTestDataset(batch_size=1, is_shuffle=False)

    optimizer = optimizers.SGD(learning_rate=0.001)
    lossFunction = lossFunctions.SparseCrossEntropy(class_num=10)

    model = validateModels.FAmodel(optimizer=optimizer, lossFunction=lossFunction)



    EPOCH = 10
    BATCH_SIZE = 64

    total_train_losses = []
    total_test_losses = []
    for e in range(EPOCH):
        print(f'EPOCH ({e+1}/{EPOCH})')
        train_x, train_y = dataset.loadTrainDataset(batch_size=BATCH_SIZE, is_shuffle=True)

        train_losses = model.train(x=train_x, y=train_y)
        test_losses = model.inference(x=test_x, y=test_y)
        print()

        test_loss = sum(test_losses) / len(test_losses)

        total_train_losses.extend(train_losses)
        total_test_losses.append(test_loss)

    
    historyVisualizer.visualize(
        train_losses={'BP': total_train_losses},
        test_losses={'BP': total_test_losses},
        epoch=EPOCH
    )

    