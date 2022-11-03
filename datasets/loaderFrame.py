import numpy as np

class LoaderFrame:
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