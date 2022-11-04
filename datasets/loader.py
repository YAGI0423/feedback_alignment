from abc import abstractmethod, ABCMeta

import pickle
import numpy as np

class LoaderFrame:
    @abstractmethod
    def _readDataset(self):
        # 인스턴스 생성 시, 데이터셋 생성 또는 불러오기
        pass

    def _shuffle_dataset(self, x, y):
        dataset_size = x.shape[0]

        shuffle_idx = list(range(dataset_size))
        np.random.shuffle(shuffle_idx)
        return x[shuffle_idx], y[shuffle_idx]

    def _split_dataset(self, x, batch_size):
        dataset_size = x.shape[0]

        split_size = int(dataset_size / batch_size)
        x = np.array_split(x, split_size)
        return x

    def loadTrainDataset(self, batch_size: int=1, is_shuffle: bool=False):
        x, y = self._x_train.copy(), self._y_train.copy()

        if is_shuffle:
            x, y = self._shuffle_dataset(x=x, y=y)

        x = self._split_dataset(x=x, batch_size=batch_size)
        y = self._split_dataset(x=y, batch_size=batch_size)
        return x, y

    def loadTestDataset(self, batch_size: int=1, is_shuffle: bool=False):
        x, y = self._x_test.copy(), self._y_test.copy()

        if is_shuffle:
            x, y = self._shuffle_dataset(x=x, y=y)
        
        x = self._split_dataset(x=x, batch_size=batch_size)
        y = self._split_dataset(x=y, batch_size=batch_size)
        return x, y

class LinearFunctionApproximation(LoaderFrame):
    '''
    Task (1) 'Linear function approximation'에 해당하는 데이터셋
    '''
    def __init__(self, input_shape, output_shape, train_dataset_size):
        (self._x_train, self._y_train), (self._x_test, self._y_test) = \
            self._readDataset(input_shape, output_shape, train_dataset_size)

    def _readDataset(self, input_shape, output_shape, train_dataset_size):
        total_dataset = int(train_dataset_size * 1.25)
        X = np.random.normal(
            loc=0, #mean
            scale=train_dataset_size, #deviation distribution
            size=(total_dataset, input_shape)
        )
        T = np.random.rand(input_shape, output_shape)
        Y = np.matmul(X, T)

        x_train, x_test = X[:train_dataset_size], X[train_dataset_size:]
        y_train, y_test = Y[:train_dataset_size], Y[train_dataset_size:]
        return (x_train, y_train), (x_test, y_test)

class Mnist(LoaderFrame):
    '''
    Task (2) 'MNIST dataset'에 해당하는 데이터셋
    '''
    def __init__(
        self,
        is_normalize: bool=False,
        is_one_hot: bool=False,
        path: str='./datasets/mnist_dataset.pk'):

        (self._x_train, self._y_train), (self._x_test, self._y_test) = \
            self._readDataset(path=path, is_normalize=is_normalize, is_one_hot=is_one_hot)
    
    def __sparse_to_oneHot(self, y):
        y = y.reshape(-1)
        return np.eye(self.class_num)[y]

    def __normalize(self, x):
        '''
        데이터셋을 -1. ~ +1. 사이의 값으로 정규화하여 반환
        '''
        return 2. * (x / 255.) - 1

    def _readDataset(self, path: str, is_normalize: bool=False, is_one_hot: bool=False):
        with open(path, 'rb') as fr:
            mnist_dataset = pickle.load(fr)

        x_train, y_train = mnist_dataset['x_train'], mnist_dataset['y_train']
        x_test, y_test = mnist_dataset['x_test'], mnist_dataset['y_test']

        if is_one_hot:
            y_train = self.__sparse_to_oneHot(y_train)
            y_test = self.__sparse_to_oneHot(y_test)

        if is_normalize:
            x_train = self.__normalize(x_train)
            x_test = self.__normalize(x_test)

        return (x_train, y_train), (x_test, y_test)