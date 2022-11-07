#<Network builder>
from ann import layers
from ann.weightInitializers import Inintializers
from ann.models import Model

#<Network compile>
from ann import lossFunctions
from ann import optimizers

#<Dataset>
from datasets import loader

#<Visualizer>
from plot import historyVisualizer

from tqdm import tqdm

import numpy as np

class TaskOneModel(Model):
    '''
    ann.models.Model 클래스를 상속받아
    논문의 Task (2)에 적합한 메소드를 추가하여 재정의한 Model 생성 클래스
    '''
    def update_on_epoch(self, x, y):
        '''
        배치 단위로 구분된 데이터셋 쌍 x, y를 입력받아,
        네트워크 epoch 1회 갱신 후, loss 리스트 및 Δh_BP 또는 Δh_FA 리스트 반환
        '''
        dataset_iter = tqdm(zip(x, y), total=len(x))

        losses, dhs = [], [] #dhs: Δh list
        for x, y in dataset_iter:
            y_hat = self.predict(x=x)
            loss = self.lossFunction.forwardProp(y_hat=y_hat, y=y)

            dLoss = self.lossFunction.backProp()

            #Δh_BP = transpose(W)·e, Δh_FA = B·e
            dh = self.output_layer.backProp(dy=dLoss)

            self.update_on_batch(dLoss)
            dataset_iter.set_description(f'Loss: {loss:.5f}')
            dh = np.mean(dh, axis=0)

            losses.append(loss)
            dhs.append(dh)
        return losses, dhs

    def inference(self, x, y):
        '''
        배치 단위로 분리된 데이터셋 쌍 x, y를 입력받아 추론(inference) 후,
        loss의 변화에 대한 리스트 반환
        '''
        dataset_iter = tqdm(zip(x, y), total=len(x))
        
        losses = []
        for x, y in dataset_iter:
            y_hat = self.predict(x=x)
            loss = self.lossFunction.forwardProp(y_hat=y_hat, y=y)

            dataset_iter.set_description(f'Loss: {loss:.5f}')
            losses.append(loss)
        return losses

    def update_network(self, dataset, epoch: int, batch_size: int):
        '''
        입력받은 `dataset`을 `batch_size` 단위로 나누어 `epoch` 만큼 네트워크를 갱신한 후,
        학습 데이터셋과 테스트 데이터셋에 대한 Loss 변화 리스트를 반환 
        '''
        test_x, test_y = dataset.loadTestDataset(batch_size=1, is_shuffle=False)

        total_train_losses = []
        total_test_losses = []
        total_dh_list = []
        for e in range(epoch):
            print(f'EPOCH ({e+1}/{epoch})')
            train_x, train_y = dataset.loadTrainDataset(batch_size=batch_size, is_shuffle=True)

            train_losses, dhs = self.update_on_epoch(x=train_x, y=train_y)
            test_losses = self.inference(x=test_x, y=test_y)
            print()

            test_loss = sum(test_losses) / len(test_losses)

            total_train_losses.extend(train_losses)
            total_test_losses.append(test_loss)
            total_dh_list.extend(dhs)
        return total_train_losses, total_test_losses, total_dh_list

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
    model = TaskOneModel(inputs=inputs, outputs=out)
    return model

if __name__ == '__main__':
    EPOCH = 2
    BATCH_SIZE = 8
    LEARNING_RATE = 0.005

    dataset = loader.LinearFunctionApproximation(train_dataset_size=25000, input_shape=30, output_shape=10, is_normalize=True)

    bp_model = create_network(affine_type='BP')
    fa_model = create_network(affine_type='FA')

    bp_model.compile(lossFunction=lossFunctions.SE(), optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))
    fa_model.compile(lossFunction=lossFunctions.SE(), optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))

    bp_train_his, bp_test_his, h_BPs = bp_model.update_network(dataset=dataset, epoch=EPOCH, batch_size=BATCH_SIZE)
    fa_train_his, fa_test_his, h_FAs = fa_model.update_network(dataset=dataset, epoch=EPOCH, batch_size=BATCH_SIZE)

    print(h_BPs)
    exit()
    
    historyVisualizer.visualize(
        path='./plot/images/task1_linearFunction.png',
        train_losses={'BP': bp_train_his, 'FA': fa_train_his},
        test_losses={'BP': bp_test_his, 'FA': fa_test_his},
        epoch=EPOCH, tick_step=100
    )