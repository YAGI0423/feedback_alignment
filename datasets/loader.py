from abc import abstractmethod, ABCMeta

import pickle
import numpy as np

class LoaderFrame(metaclass=ABCMeta):
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

    목표 선형 함수 𝑇는 30차원 공간의 벡터를 10차원으로 매핑하였다.
    𝑇의 요소는 무작위로, [−1, 1] 범위로부터 균일하게 추출되었다.
    데이터셋 𝐷 = {(𝑥1, 𝑦1), ⋯ (𝑥𝑁, 𝑦𝑁)}는 𝑥𝑖 ~ 𝑁(𝜇 = 0, ∑ = 𝐼)인,𝑦𝑖 = 𝑇𝑥𝑖에 따라 생성되었다.
    (Full Methods 참조)
    '''
    def __init__(self, train_dataset_size, input_shape=30, output_shape=10):
        (self._x_train, self._y_train), (self._x_test, self._y_test) = \
            self._readDataset(input_shape, output_shape, train_dataset_size)

    def _readDataset(self, input_shape, output_shape, train_dataset_size):
        total_dataset = int(train_dataset_size * 1.25)
        X = np.random.normal(
            loc=0, #mean
            scale=train_dataset_size, #deviation distribution
            size=(total_dataset, input_shape)
        )
        T = np.random.rand(input_shape, output_shape) * 2 - 1
        Y = np.matmul(X, T)

        x_train, x_test = X[:train_dataset_size], X[train_dataset_size:]
        y_train, y_test = Y[:train_dataset_size], Y[train_dataset_size:]
        return (x_train, y_train), (x_test, y_test)

class Mnist(LoaderFrame):
    '''
    Task (2) 'MNIST dataset'에 해당하는 데이터셋

    네트워크는 0-9의 필기 숫자 이미지를 분류하도록 학습되었다.
    표준 원-핫 표현은 원하는(desired) 출력을 코딩하는 데 사용되었다.

    네트워크는 기본 MNIST 데이터셋[17] 60,000개의 이미지로 학습되었다.
    그리고 성능은 10,000개의 이미지 테스트 셋에서 발생한 오차의 백분율로 측정되었다.
    (Methods Summary 참조)

    ※ MNIST dataset은 제공하지 않는다 ※
    '''
    def __init__(
        self,
        is_normalize: bool=False,
        is_one_hot: bool=False,
        path: str='./datasets/mnist_dataset.pk'):

        (self._x_train, self._y_train), (self._x_test, self._y_test) = \
            self._readDataset(path=path, is_normalize=is_normalize, is_one_hot=is_one_hot)
    
    def __sparse_to_oneHot(self, y):
        CLASS_NUM = 10
        y = y.reshape(-1)
        return np.eye(CLASS_NUM)[y]

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

class NonlinearFunctionApproximation(LoaderFrame):
    '''
    Task (3) 'Noninear function approximation'에 해당하는 데이터셋

    데이터셋 𝐷 = {(𝑥1, 𝑦1), ⋯ (𝑥𝑁, 𝑦𝑁)}는 𝑥𝑖 ~ 𝑁(𝜇 = 0, ∑ = 𝐼)인, 𝑦𝑖 = 𝑇(𝑥𝑖)에 따라 생성되었다.
    '''
    def __init__(self, train_dataset_size, input_shape=30, output_shape=10):
        (self._x_train, self._y_train), (self._x_test, self._y_test) = \
            self._readDataset(input_shape, output_shape, train_dataset_size)

    def _readDataset(self, input_shape, output_shape, train_dataset_size):
        from datasets import thirdTaskTargetNet
        T = thirdTaskTargetNet.getNetwork(input_shape=input_shape, output_shape=output_shape)

        total_dataset = int(train_dataset_size * 1.25)
        X = np.random.normal(
            loc=0, #mean
            scale=train_dataset_size, #deviation distribution
            size=(total_dataset, input_shape)
        )
        Y = T.predict(x=X)

        x_train, x_test = X[:train_dataset_size], X[train_dataset_size:]
        y_train, y_test = Y[:train_dataset_size], Y[train_dataset_size:]
        return (x_train, y_train), (x_test, y_test)