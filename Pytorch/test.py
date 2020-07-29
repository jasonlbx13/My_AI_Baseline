import sys
import os
import scipy.io as sio
import cv2
import torch
import time
import numpy as np
import onnx
import onnxruntime
import MNN
import torch.nn.functional as F
import torch.nn as nn
# from model.dbface_light import DBFace
from model.dbface_small import DBFace
from utils import preprocess
from model.losses import GIoULoss
from tqdm import tqdm


mean = [0.408, 0.447, 0.47]
std = [0.289, 0.274, 0.278]
label_file = '/home/data/Datasets/SD/self_test/label.txt'
images_dir = '/home/data/Datasets/SD/self_test/images/'
giou_loss = GIoULoss()

def eval_reg(model, gt):
    imgfile, objs = gt
    img = cv2.imread(imgfile)
    H, W, _ = img.shape
    mhw = max(H, W)
    scale = mhw / 256
    new_img = np.zeros((mhw, mhw, 3), np.float32)
    reg_tlrb = np.zeros((1 * 4, 64, 64), np.float32)
    reg_mask = np.zeros((1, 64, 64), np.float32)
    new_img[:H, :W, :] = img
    img = cv2.resize(new_img, (256, 256))

    for obj in objs:
        obj.x = obj.x / scale
        obj.y = obj.y / scale
        obj.r = obj.r / scale
        obj.b = obj.b / scale
        cx, cy = obj.safe_scale_center(1 / 4, 64, 64)
        reg_box = np.array(obj.box) / 4
        reg_tlrb[:, cy, cx] = reg_box
        reg_mask[0, cy, cx] = 1

    reg_tlrb = torch.tensor(reg_tlrb).unsqueeze(0).cuda()
    reg_mask = torch.tensor(reg_mask).unsqueeze(0).cuda()
    img = ((img / 255. - mean) / std).astype(np.float32)
    img = cv2.resize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)[None].cuda()
    hm, pred_tlrb, _ = model(img)
    reg_loss = giou_loss(pred_tlrb, reg_tlrb, reg_mask)
    return reg_loss

if __name__ == '__main__':

    model = DBFace(has_landmark=True, wide=64, has_ext=True, upmode="UCBA")
    model.load('./model/model_file/dbface_nearsmall_rubust_selfdata2.pth')
    model.eval()
    model.cuda()

    test_gt = preprocess.load_webface(label_file, images_dir)
    reg_loss = []
    for i in tqdm(range(len(test_gt))):
        reg_loss.append(eval_reg(model, test_gt[i]).data.cpu().numpy())

    score = 1-np.array(reg_loss).mean()
    print (score)