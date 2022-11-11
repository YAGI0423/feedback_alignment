from ann.models import Model

from tqdm import tqdm

class TaskModel(Model):
    '''
    ann.models.Model 클래스를 상속받아
    논문에 적합한 메소드를 추가하여 재정의한 Model 생성 클래스
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