import numpy as np

class Inintializers:
    @staticmethod
    def randomUniform(*args):
        return np.random.uniform(-1., 1., size=args)

    @staticmethod
    def randomNormal(*args):
        return np.random.randn(*args)

    @staticmethod
    def Xavier(*args):
        return np.random.randn(*args) * np.sqrt(2. / np.sum(args))

    @staticmethod
    def He(*args):
        n_in = args[0]
        return np.random.randn(*args) * np.sqrt(2. / n_in)

    @staticmethod
    def TaskInit(*args):
        '''
        Task (1), (3)에서 시행한 초기화 방법

        네트워크 가중치 행렬의 요소, 𝑊0, 𝑊는 범위 [−0.01, 0.01]에서 균일하게 추출하여 초기화 되었다.
        (Full Methods 참조)
        '''
        return np.random.uniform(-0.01, 0.01, size=args)