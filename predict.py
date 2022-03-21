import numpy as np
import torch
import torch.nn as nn
import cv2
from glob import glob

def result_img(pred, img):
    for i in range(0, 384):
        for j in range(0, 384):
            if pred[i][j] != 0:
                img[i][j][1] = 255
    return img

def predict(model, imgs):
    model.eval()
    i = 0
    for img in imgs:
        imgnp = cv2.imread(img)
        imgnp = cv2.resize(imgnp, (384, 384), interpolation=cv2.INTER_CUBIC)
        imgnp_2 = imgnp[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype("float32")
        input = torch.from_numpy(imgnp_2).cuda()
        with torch.no_grad():
            output = model(input)
            output = output.squeeze()
        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float()
        prednp = np.array(pred.cpu())
        res_img = result_img(prednp, imgnp)
        i += 1

        cv2.imwrite('./output/{}.png'.format(i), res_img)

    return



if __name__ == "__main__":

    # Configuration Adjustment
    load_path = './pretrain/RAT_t3.pth' # .pth file path
    imgs = glob(r'./pictures/*') # test image file path



    from model.RAtransformer import RAT
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

    predict(model, imgs)

