from abc import abstractmethod, ABCMeta
import numpy as np

class Layer(metaclass=ABCMeta):
    @abstractmethod
    def forwardProp(self, x):
        pass

    @abstractmethod
    def backProb(self, dy):
        pass

class LossFunction(metaclass=ABCMeta):
    @abstractmethod
    def forwardProp(self, y_hat, y):
        pass

    @abstractmethod
    def backProb(self):
        pass

class BPLayer(Layer):
    def __init__(self, input_shape, units):
        self.W = np.random.rand(input_shape, units)
        self.b = np.random.rand(units)

        self._rec_x = None

    def forwardProp(self, x):
        xW = np.matmul(x, self.W)
        h = xW + self.b

        self._rec_x = x
        return h

    def backProb(self, dy):
        dx = np.dot(dy, self.W.T)

        #<Affine Method>
        db = dy.sum(axis=0)
        xT = np.transpose(self._rec_x)
        dW = np.dot(xT, dy)
        return dx, dW, db

        #<Personal Method>
        '''
        보편적인 방법에 해당하는 『Affine Method』의 경우,
        배치 별 가중치 미분 값을 합하여 출력한다.
        『Personal Methd』의 경우 배치에 따른 가중치를 그대로 출력한다.
        '''
        db = dy
        dW = np.multiply(dy, self._rec_x)
        return dx, dW, db

class Sigmoid(Layer):
    def __init__(self):
        self._rec_o = None

    def forwardProp(self, x):
        o = 1 / (1 + np.exp(-x))
        
        self._rec_o = o
        return o

    def backProb(self, dy):
        do = self._rec_o * (1 - self._rec_o)
        return dy * do

class MSE(LossFunction):
    def __init__(self):
        self._rec_err = None

    def forwardProp(self, y_hat, y):
        err = np.subtract(y_hat, y)
        loss = np.square(err)
        loss = np.mean(loss)

        self._rec_err = err
        return loss

    def backProb(self):
        batch_by_class = np.prod(self._rec_err.shape) #batch size * class num
        dloss = (2 / batch_by_class) * self._rec_err
        
        #<Batch x class로 나누는 이유에 대해>
        '''
        손실함수 단계에서 미리 미분값을 배치 크기로 나누면,
        이후 Affine 계층의 ∂L/∂W 값을 matmul(Δy, Δx.T) 만으로 배치 크기를 고려하여 구할 수있다.
        해당 연산이 없는 경우, 동일한 미분값을 얻기위해서는 각 ΔW를 배치 크기로 나누어주어야 하는 번거로움이 있다. 
        '''
        return dloss



if __name__ == '__main__':
    faLayer = BPLayer(input_shape=2, units=1)
    sigmoid = Sigmoid()
    loss_func = MSE()

    print('W: ', faLayer.W, end='\n\n')
    print('b: ', faLayer.b, end='\n\n')

    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [1]]

    print('x: ', x, end='\n\n')
    print('y: ', y, end='\n\n')

    h = faLayer.forwardProp(x=x)
    print('h: ', h, end='\n\n')

    o = sigmoid.forwardProp(x=h)
    print('o: ', o, end='\n\n')

    loss = loss_func.forwardProp(y_hat=o, y=y)
    print('loss: ', loss, end='\n\n')

    print('=' * 50)

    do = loss_func.backProb()
    print('do: ', do, end='\n\n')

    dh = sigmoid.backProb(dy=do)
    print('dh: ', dh, end='\n\n')

    dx, dW, db = faLayer.backProb(dy=dh)
    print('dx: ', dx, end='\n\n')
    print('dW: ', dW, end='\n\n')
    print('db: ', db, end='\n\n')