#<Network builder>
from ann import layers
from ann.weightInitializers import Inintializers
from ann.models import Model

#<Network compile>
from ann import lossFunctions
from ann import optimizers

def create_network(affine_type: str='BP'):
    '''
    `Task (1) Linear function approximation`에서 사용한 모델 반환

    30-20-10 선형 네트워크가 선형 함수, 𝑇를 근사하도록 학습하였다.
    𝐵는 균일 분포 [−0.5, 0.5]에서 추출된다(=TaskInit).
    (Methods Summary 참조)
    '''
    if affine_type == 'BP': #Back Propagation
        AffineLayer = layers.BPLayer
    elif affine_type == 'FA': #Feedback Alignment
        AffineLayer = layers.FALayer
    else: #except
        raise  Exception('\n\n\nThe parameter `affine_type` must be "BP" or "FA" in string format.\n\n')

    inputs = layers.InputLayer(shape=(30, ))
    out = AffineLayer(input_shape=30, units=20, weight_init=Inintializers.TaskInit)(inputs)
    out = AffineLayer(input_shape=20, units=10, weight_init=Inintializers.TaskInit)(out)
    model = Model(inputs=inputs, outputs=out)
    return model