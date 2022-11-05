from ann import layers
from ann.models import Model
from ann.weightInitializers import Inintializers

'''
`Task (3) Nonlinear function approximation`에서 사용되는 목표 네트워크(Target Network), T(·)를 반환하는 모듈

매개변수는 무작위로 선택되고 대응하는 모든 시뮬레이션에 대해 고정된다.
데이터셋 𝐷 = {(𝑥1, 𝑦1), ⋯ (𝑥𝑁, 𝑦𝑁)}는 𝑥𝑖 ~ 𝑁(𝜇 = 0, ∑ = 𝐼)으로, 𝑦𝑖 = 𝑇𝑥𝑖에 따라 생성되었다.
(Full Methods 참조)
'''

def getNetwork(input_shape, output_shape):
    '''
    목표 네트워크는 tanh(·)인 은닉, 출력 유닛을 가지고 각 유닛은 편향을 가진다.
    입/출력 학습 쌍은 y = W2 · tanh · ⁡(W1tanh · ⁡(W0·x + b0) + b1) + b2을 통해 생성되었다.
    (Methods Summary 참조)
    '''
    inputs = layers.InputLayer(shape=(input_shape, ))
    out = layers.BPLayer(input_shape=input_shape, units=20, weight_init=Inintializers.randomUniform)(inputs)
    out = layers.Tanh()(out)

    out = layers.BPLayer(input_shape=20, units=10, weight_init=Inintializers.randomUniform)(out)
    out = layers.Tanh()(out)

    out = layers.BPLayer(input_shape=10, units=output_shape, weight_init=Inintializers.randomUniform)(out)

    model = Model(inputs=inputs, outputs=out)
    return model