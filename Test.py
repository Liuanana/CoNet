import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.CoNet import CoNet
from utils.dataloader1 import test_dataset
import cv2
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snaps/CoNet/CoNet-54.pth')

for _data_name in ['GEDD']:
    data_path = './data/Test/{}/'.format(_data_name)
    save_path = './results_55_new/CoNet/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = CoNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        # ？
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5, res4, res3, res2, res6, res7, res8, res9, res1 = model(image)
        res = res1
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        # ？
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        

        misc.imsave(save_path + name, res)
