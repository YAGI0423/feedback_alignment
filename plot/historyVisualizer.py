import matplotlib.pyplot as plt

def __pltDefaultSetting(plt, title: str, ticks, labels):
    title_args = {'fontsize': 13, 'fontweight': 'bold'}

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
    __pltDefaultSetting(plt, title='Loss on Train Set', ticks=ticks, labels=labels)

def __pltTestLoss(plt, test_losses: dict, epoch: int, tick_step: int, color_list: list):
    for (name, losses), color in zip(test_losses.items(), color_list):
        plt.plot(losses, color=color, label=name)
    plt.legend(loc='upper right')

    ticks = list(range(-1, epoch+1, tick_step))
    ticks[0] = 0
    labels = list(range(0, epoch+1, tick_step))
    labels[0] = 1
    __pltDefaultSetting(plt, title='Loss on Test Set', ticks=ticks, labels=labels)
    

def firSecTaskVisualize(path: str, title: str, train_losses: dict, test_losses: dict, epoch: int, tick_step: int):
    LOSS_COLOR_LIST = ['#000000', '#00AF00'] #BP, FA

    plt.figure(figsize=(14, 5.5))
    plt.suptitle(title, fontsize=15, fontweight ='bold')
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.125, top=0.825, wspace=0.175, hspace=0.1)

    #Train History===========================
    plt.subplot(1, 2, 1)
    __pltTrainLoss(plt, train_losses=train_losses, epoch=epoch, tick_step=tick_step, color_list=LOSS_COLOR_LIST)
    plt.xlabel('Epochs')
    #End=====================================

    #Test History============================
    plt.subplot(1, 2, 2)
    __pltTestLoss(plt, test_losses=test_losses, epoch=epoch, tick_step=tick_step, color_list=LOSS_COLOR_LIST)
    plt.xlabel('Epochs')
    #End=====================================

    plt.savefig(path)
    plt.show()