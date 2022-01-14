import os
import sys

import numpy as np
import time
import random
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from glob import glob

import losses as Loss

from utils.dataset_RAT import Dataset
from torch.utils.data import DataLoader, random_split
from metrics import fast_hist, Dice


load_pretrain = False
load_my = False
load_path = './Log/RAT_t3/RAT_t3.pth'
mymodel_name = "RAT_t3"
BATCH_SIZE = 12
EPOCH = 100
LEARNING_RATE = 0.0001

loss_names = list(Loss.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

current_filename = 'Log/' + mymodel_name + time.strftime('%Y-%m-%d-%H-%M')
log_file = os.path.join(current_filename, 'log.txt')
lr_file = os.path.join(current_filename, 'lr.txt')
mymodel_file = os.path.join(current_filename, mymodel_name + ".pth")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, model, fo):
    model.eval()
    i = 0
    dice_whole = np.zeros(len(val_loader))
    for (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
        input = input.cuda()
        target = target.cuda()
        target = target.squeeze()
        with torch.no_grad():
            output = model(input)
            output = output.squeeze()
        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float()

        hist = fast_hist(target, pred)
        hist_np = hist.cpu().numpy()
        dice_whole[i] = Dice(hist_np, pred, target).cpu().numpy()
        i += 1
    dice_whole = np.sort(dice_whole)
    print(
        f"\nDice:{np.mean(dice_whole)}")
    # 添加loss
    fo.write(
        '\n' + f"Dice:{np.mean(dice_whole)}" + '\n')
    fo.write(
        '\n' + f"Dice_mm:{np.std(dice_whole)}, " + '\n')
    fo.flush()

    return np.mean(dice_whole)


def train(
        model,
        epochs=5,
        batch_size=32,
        lr=0.00001,
):
    # 添加1
    if not os.path.lexists(current_filename):
        os.mkdir(current_filename)
    fo = open(log_file, "a+")
    fo.write("Open the log!\n")
    fo.write("train time:" + current_filename + '\n')
    fo.write("model:" + mymodel_name + '\n')

    flr = open(lr_file, "a+")
    fo.flush()
    flr.flush()
    print("open file!")

    # 写入基本信息：
    fo.write(f"batch_size={batch_size}" + '\n')
    fo.write(f"epochs={epochs}" + '\n')
    fo.write(f"learn_rate:{lr}" + '\n')
    fo.flush()

    img_paths = glob(r'/home/train_img/*')    # train_data path
    label_paths = glob(r'/home/train_label/*')     # train_label path
    valimg_paths = glob(r'/home/val_img/*')   # val_data path
    vallabel_paths = glob(r'/home/val_label/*')   # val_label path

    train_dataset = Dataset(img_paths, label_paths, train_flag=True)
    val_dataset = Dataset(valimg_paths, vallabel_paths, train_flag=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=3, pin_memory=True, drop_last=True)

    criterion = Loss.BCEDiceLoss().cuda()

    fo.write(f"criterion:BCEDiceLoss()" + '\n')
    fo.flush()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=6)
    fo.write(f"optim:Adam" + '\n')
    fo.flush()

    best = 0.0
    global_step = 0
    for epoch in range(epochs):
        model.train()
        fo.write('\n' + f"-------------------epoch={epoch}-------------------" + '\n')
        fo.flush()
        print("LR is {}".format(optimizer.param_groups[0]['lr']))
        flr.write(f"lr:{optimizer.param_groups[0]['lr']}" + '\n')
        flr.flush()
        losses = AverageMeter()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (input, target) in pbar:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            output = output.squeeze()
            target = target.squeeze()
            loss = criterion(output, target)
            pbar.set_postfix(**{'loss(batch)': loss.item()})
            losses.update(loss.item(), input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if (global_step) % int(len(train_loader) // 2) == 0 and epoch % 1 == 0:
                fo.write(f"lr:{optimizer.param_groups[0]['lr']}" + '\n')
                fo.flush()
                flr.write(f"lr:{optimizer.param_groups[0]['lr']}" + '\n')
                flr.flush()
                fo.write('\n' + f"losses.avg:{losses.avg}" + '\n')
                fo.flush()

                dice_score = validate(val_loader, model, fo)

                if (dice_score > best):
                    torch.save(model.state_dict(), mymodel_file)
                    print('Saved the best model！')
                    best = dice_score
                mylastmodel_file = os.path.join(current_filename, time.strftime('%H-%M') + ".pth")
                torch.save(model.state_dict(), mylastmodel_file)
                fo.write('\n' + f">>>  Best_sorce:{best}, Epoch:{epoch}" + '\n')
                fo.flush()
                scheduler.step(dice_score)

        print(f"\nlosses.avg:{losses.avg}" + '\n')
        torch.cuda.empty_cache()
    fo.write('\n' + "-------------------model------------------" + '\n')
    fo.flush()
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
        fo.write('\n' + name + ':' + f"{parameters.size()}" + '\n')
        fo.write('\n' + f"max_weight:{parameters.detach()[0].max()}" + '\n')
        fo.flush()


if __name__ == "__main__":
    setup_seed(20)

    print("Learning rate is {}".format(LEARNING_RATE))
    from model.RAtransformer import RAT

    model = RAT(num_classes=1, threshold='t3')

    torch.cuda.set_device(0)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
    # device_num = 'cuda:3'
    # device = torch.device(device_num if torch.cuda.is_available() else 'cpu')
    # model.to(device=device)

    if load_pretrain:
        print('Load pretrain model!')
        pretrained_dict = torch.load(load_path)
        model_dict = model.state_dict()
        qz = 'module.encoder.'
        pretrained_dict = {(qz + k): v for k, v in pretrained_dict.items() if (qz + k) in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if load_my:
        print('Load My pretrain model!')
        pretrained_dict = torch.load(load_path)
        model.load_state_dict(pretrained_dict, strict=False)

    try:
        train(model, EPOCH, BATCH_SIZE, LEARNING_RATE)
    except KeyboardInterrupt:
        model_interrupted_file = os.path.join(current_filename, "INTERRUPTED.pth")
        torch.save(model.state_dict(), model_interrupted_file)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
