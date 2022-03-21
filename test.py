import numpy as np
import torch
from tqdm import tqdm
from glob import glob

from model.RAtransformer import RAT
from utils.dataset_RAT import Dataset
from torch.utils.data import DataLoader, random_split


# There are many verification metrics in the metrics file, and you can choose the verification metric you want to add.
from metrics import fast_hist, Dice



def test(test_loader, model):
    model.eval()
    dice_whole = np.zeros(len(test_loader))
    for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
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
    dice_whole = np.sort(dice_whole)
    print(
        f"\nDice:{np.mean(dice_whole)}")

    return



if __name__ == "__main__":

    # Configuration Adjustment
    load_path = './pretrain/RAT_t3.pth' # .pth file path
    testimg_paths = glob(r'/home/testimg/*') # test image file path
    testlabel_paths = glob(r'/home/testlabel/*') # label file path


    model = RAT(num_classes=1, threshold='t3')
    device_num = 'cuda:0'
    device = torch.device(device_num if torch.cuda.is_available() else 'cpu')
    model.to(device=device)

    pretrained_dict = torch.load(load_path)
    model_dict = model.state_dict()
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    pretrained_dict_new = {(k): v for k, v in new_state_dict.items() if (k) in model_dict}
    model_dict.update(pretrained_dict_new)
    model.load_state_dict(model_dict)

    test_dataset = Dataset(testimg_paths, testlabel_paths, train_flag=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=3, pin_memory=True, drop_last=True)

    test(model, test_loader)

