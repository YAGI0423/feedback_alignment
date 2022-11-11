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

class TaskThreeModel(Model):
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

        losses = []
        for x, y in dataset_iter:
            y_hat = self.predict(x=x)
            loss = self.lossFunction.forwardProp(y_hat=y_hat, y=y)

            dLoss = self.lossFunction.backProp()

            self.update_on_batch(dLoss)
            dataset_iter.set_description(f'Loss: {loss:.5f}')

            losses.append(loss)
        return losses

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
        for e in range(epoch):
            print(f'EPOCH ({e+1}/{epoch})')
            train_x, train_y = dataset.loadTrainDataset(batch_size=batch_size, is_shuffle=True)

            train_losses = self.update_on_epoch(x=train_x, y=train_y)
            test_losses = self.inference(x=test_x, y=test_y)
            print()

            test_loss = sum(test_losses) / len(test_losses)

            total_train_losses.extend(train_losses)
            total_test_losses.append(test_loss)
        return total_train_losses, total_test_losses

def create_network(affine_type: str='BP', add_layer: bool=False):
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
    out = AffineLayer(input_shape=30, units=20, weight_init=Inintializers.Xavier)(inputs)
    out = layers.Tanh()(out)

    out = AffineLayer(input_shape=20, units=10, weight_init=Inintializers.TaskInit)(out)

    if add_layer: #30-20-10-10 Network
        out = layers.Tanh()(out)
        out = AffineLayer(input_shape=10, units=10, weight_init=Inintializers.TaskInit)(out)

    model = TaskThreeModel(inputs=inputs, outputs=out)
    return model

def moving_average(x, window=10):
        x = list(x).copy()
        iter_num = len(x) - window - 1

        averages = []
        for _ in range(iter_num):
            mean = sum(x[:window]) / window
            averages.append(mean)
            del x[0]
        return averages

if __name__ == '__main__':
    EPOCH = 10
    BATCH_SIZE = 4
    LEARNING_RATE = 0.1

    dataset = loader.NonlinearFunctionApproximation(train_dataset_size=25000, input_shape=30, output_shape=10, is_normalize=True)

    networks = {
        'bp2_model': create_network(affine_type='BP', add_layer=False),
        'fa2_model': create_network(affine_type='FA', add_layer=False),
        'bp3_model': create_network(affine_type='BP', add_layer=True),
        'fa3_model': create_network(affine_type='FA', add_layer=True)
    }
    
    train_losses, test_losses = dict(), dict()
    for name, net in networks.items():
        net.compile(lossFunction=lossFunctions.SE(), optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))
        train_loss, test_losses[name] = net.update_network(dataset=dataset, epoch=EPOCH, batch_size=BATCH_SIZE)

        train_losses[name] = moving_average(train_loss, window=500)

    historyVisualizer.visualize(
        title='Task (3) Nonlinear function approximation',
        path='./plot/images/task3_nonlinearFunction.png',
        train_losses=train_losses,
        test_losses=test_losses,
        epoch=EPOCH, tick_step=5,
        test_Ylim=(min(test_losses['fa3_model'])-0.001, max(test_losses['fa3_model'])+0.001)
    )