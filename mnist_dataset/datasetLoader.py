import pickle
import numpy as np

class Loader:
    def __init__(self, path: str='./mnist_dataset/mnist_dataset.pk'):
        self.path = path

    def loadDataset(self, batch_size: int, is_normalize: bool=False, is_one_hot: bool=False):
        with open(self.path, 'rb') as fr:
            mnist_dataset = pickle.load(fr)

        x_train, y_train = mnist_dataset['x_train'], mnist_dataset['y_train']
        x_test, y_test = mnist_dataset['x_test'], mnist_dataset['y_test']

        if is_one_hot:
            y_train = self._sparse_to_oneHot(y_train)
            y_test = self._sparse_to_oneHot(y_test)

        if is_normalize:
            x_train = self._normalize(x_train)
            x_test = self._normalize(x_test)

        x_train, y_train = self._split_dataset(x_train, batch_size), self._split_dataset(y_train, batch_size)
        x_test, y_test = self._split_dataset(x_test, batch_size), self._split_dataset(y_test, batch_size)

        return (x_train, y_train), (x_test, y_test)

    def _sparse_to_oneHot(self, y):
        y = y.reshape(-1)
        return np.eye(self.class_num)[y]

    def _split_dataset(self, x, batch_size):
        dataset_size = x.shape[0]
        split_size = int(dataset_size / batch_size)
        x = np.array_split(x, split_size)
        return np.array(x)

    def _normalize(self, x):
        '''
        데이터셋을 -1. ~ +1. 사이의 값으로 정규화하여 반환
        '''
        return 2. * (x / 255.) - 1
