from abc import *

class Layer:
    @abstractmethod
    def forwardProp(self):
        pass

    @abstractmethod
    def backProb(self):
        pass

class FALayer(Layer):
    def __init__(self):
        pass

    def forwardProp(self):
        pass

    def backProb(self):
        pass


if __name__ == '__main__':
    faLayer = FALayer()
