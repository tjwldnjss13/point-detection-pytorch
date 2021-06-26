import os
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset.coco_keypoint_dataset import *
from dataset.augment import *
from models.face_shoulder_net import *
from loss import *
from utils.pytorch_util import *
from utils.util import *

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"


def get_dataset():
    root = 'D://DeepLearningData/COCO/'
    augmentation = [horizontal_flip_augmentation, shift_augmentation]
    train_dset = COCOKeypointDataset(root=root, image_size=224, for_train=True, year='2017', augmentation=augmentation)
    val_dset = COCOKeypointDataset(root=root, image_size=224, for_train=False, year='2017')
    collate_fn = custom_collate_fn

    return train_dset, val_dset, collate_fn


def adjust_learning_rate(optimizer, current_epoch):
    e = current_epoch
    if e == 0:
        optimizer.param_groups[0]['lr'] = .0001
    elif e == 5:
        optimizer.param_groups[0]['lr'] = .001
    elif e == 20:
        optimizer.param_groups[0]['lr'] = .0001


def validate(model, data_loader, device):
    model.eval()

    loss_total = 0
    num_batch = 0
    for i, (img, pnt) in enumerate(data_loader):
        num_batch += 1
        img = make_batch(img).to(device)
        tar_pnt = make_batch(pnt).to(device)

        pred_pnt = model(img)
        loss = rmse_loss(pred_pnt, tar_pnt)
        # pred_pnt = model(img)
        # loss = F.binary_cross_entropy(pred_pnt, tar_pnt)

        loss_total += loss.detach().cpu().item()

        del img, tar_pnt, pred_pnt, loss
        torch.cuda.empty_cache()

    loss = loss_total / num_batch

    return loss


def train(model, optimizer, epoch, batch_size, device):
    train_dset, val_dset, collate_fn = get_dataset()
    train_loader = DataLoader(dataset=train_dset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(dataset=val_dset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    train_loss_list = []
    val_loss_list = []

    t_start = time.time()
    for e in range(epoch):
        model.train()
        adjust_learning_rate(optimizer, e)

        loss_total = 0
        num_batch = 0
        num_data = 0
        cur_lr = optimizer.param_groups[0]['lr']
        for i, (img, pnt) in enumerate(train_loader):
            num_batch += 1
            num_data += len(img)

            print(f'[{e + 1}/{epoch}] ', end='')
            print(f'{num_data}/{len(train_dset)}  ', end='')
            print(f'<lr> {cur_lr:.6f} ', end='')

            img = make_batch(img).to(device)
            tar_pnt = make_batch(pnt).to(device)

            pred_pnt = model(img)
            # print()
            # print('pred', end=' ')
            # for p in pred_pnt[0].detach().cpu().numpy():
            #     print(f'{p:5f}', end=' ')
            # print()
            # print('tar', end='  ')
            # for p in tar_pnt[0].detach().cpu().numpy():
            #     print(f'{p:5f}', end=' ')
            # print()
            loss = rmse_loss(pred_pnt, tar_pnt)
            # pred_pnt = model(img)
            # loss = F.binary_cross_entropy(pred_pnt, tar_pnt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.detach().cpu().item()

            print(f'<loss> {loss.detach().cpu().item():.5f} ({loss_total / num_batch:.5f})  ', end='')

            t_batch_end = time.time()
            h, m, s = time_calculator(t_batch_end - t_start)
            print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}')

            del img, tar_pnt, pred_pnt, loss
            torch.cuda.empty_cache()

        train_loss = loss_total / num_batch

        train_loss_list.append(train_loss)

        with torch.no_grad():
            val_loss = validate(model, val_loader, device)
            val_loss_list.append(val_loss)
            print(f'\t\t<val_loss> {val_loss_list[-1]:.5f}')

        save_pth = f'./save/{e+1}epoch_{cur_lr:.5f}lr_{train_loss_list[-1]:.5f}loss(train)_{val_loss_list[-1]:.5f}loss(val).pth'
        torch.save(model.state_dict(), save_pth)

    plt.figure(0)
    plt.plot([i for i in range(epoch)], train_loss_list, 'r-', label='Train')
    plt.plot([i for i in range(epoch)], val_loss_list, 'b-', label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.show()


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epoch = 50
    model = FSNet().to(device)
    # model.load_state_dict(torch.load('./pretrain/50epoch_0.00010lr_2.67666loss(train)_2.79753loss(val).pth'))
    optimizer = torch.optim.SGD(model.parameters(), lr=.0001, momentum=.9, weight_decay=.0005, nesterov=True)

    train(model=model, optimizer=optimizer, batch_size=batch_size, epoch=epoch, device=device)


if __name__ == '__main__':
    main()

























