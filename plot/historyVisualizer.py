import matplotlib.pyplot as plt

def __pltDefaultSetting(plt, title: str, ticks, labels):
    title_args ={'fontsize': 13, 'fontweight': 'bold'}

    plt.title(title, fontdict=title_args, loc='left', pad=10)

    plt.xticks(ticks=ticks, labels=labels)
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')

def __pltTrainLoss(plt, train_losses: dict, epoch: int, tick_step: int, color_list: list):
    for (name, losses), color in zip(train_losses.items(), color_list):
        plt.plot(losses, color=color, label=name)
    plt.legend(loc='upper right')

    size_per_epoch = int(len(losses) / epoch) - 1
    ticks = tuple(e * size_per_epoch for e in range(0, epoch+1, tick_step))
    labels = tuple(range(0, epoch+1, tick_step))
    __pltDefaultSetting(plt, title='<Loss on Train Set>', ticks=ticks, labels=labels)

def __pltTestLoss(plt, test_losses: dict, epoch: int, tick_step: int, color_list: list):
    for (name, losses), color in zip(test_losses.items(), color_list):
        plt.plot(losses, color=color, label=name)
    plt.legend(loc='upper right')

    ticks = list(range(-1, epoch+1, tick_step))
    ticks[0] = 0
    labels = list(range(0, epoch+1, tick_step))
    labels[0] = 1
    __pltDefaultSetting(plt, title='<Loss on Test Set>', ticks=ticks, labels=labels)
    

def firTaskVisualize(path: str, train_losses: dict, test_losses: dict, angle_list: list, epoch: int, tick_step: int):
    LOSS_COLOR_LIST = ['#000000', '#00AF00'] #BP, FA

    plt.figure(figsize=(17, 5))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.87, wspace=0.25, hspace=0.15)

    #Train History===========================
    plt.subplot(1, 3, 1)
    __pltTrainLoss(plt, train_losses=train_losses, epoch=epoch, tick_step=tick_step, color_list=LOSS_COLOR_LIST)
    #End=====================================

    #Test History============================
    plt.subplot(1, 3, 2)
    __pltTestLoss(plt, test_losses=test_losses, epoch=epoch, tick_step=tick_step, color_list=LOSS_COLOR_LIST)
    #End=====================================

    def moving_average(x, window=10):
        x = list(x).copy()

        moving_averages = [None] * (window - 1)
        
        iter_num = len(x) - window + 1
        
        for _ in range(iter_num):
            value = sum(x[:window]) / window
            moving_averages.append(value)
            x.pop(0)
        return moving_averages
    
    averages = moving_average(angle_list, window=10)

    plt.subplot(1, 3, 3)
    plt.plot(angle_list, color='#AFE5AF')
    plt.plot(averages, color='#00AF00')


    size_per_epoch = int(len(angle_list) / epoch) - 1
    ticks = tuple(e * size_per_epoch for e in range(0, epoch+1, tick_step))
    labels = tuple(range(0, epoch+1, tick_step))
    __pltDefaultSetting(plt, title='<∆ℎFA ∠ ∆ℎBP on Train Set>', ticks=ticks, labels=labels)
    plt.xlabel('Epochs')

    plt.savefig(path)
    plt.show()

def secTaskVisualize(path: str, train_losses: dict, test_losses: dict, epoch: int, tick_step: int):
    LOSS_COLOR_LIST = ['#000000', '#00AF00'] #BP, FA

    plt.figure(figsize=(7, 9))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.925, wspace=0.1, hspace=0.3)

    #Train History===========================
    plt.subplot(2, 1, 1)
    __pltTrainLoss(plt, train_losses=train_losses, epoch=epoch, tick_step=tick_step, color_list=LOSS_COLOR_LIST)
    #End=====================================

    #Test History============================
    plt.subplot(2, 1, 2)
    __pltTestLoss(plt, test_losses=test_losses, epoch=epoch, tick_step=tick_step, color_list=LOSS_COLOR_LIST)
    plt.xlabel('Epochs')
    #End=====================================

    plt.savefig(path)
    plt.show()