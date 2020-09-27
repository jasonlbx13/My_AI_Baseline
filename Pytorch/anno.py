import pandas as pd
import numpy as np
import cv2
import PIL.Image as Image
import os
from tqdm import tqdm
from inference import detect
from model.dbface_big import DBFace


def get_root(label_path):
    with open(label_path, 'r') as f:
        data = f.readlines()
    name_list = []
    for line in data:
        if line[0] == '#':
            name_list.append(line[2:-1])
    del data
    return name_list


def predict(root, name_list, model):

    objss = {}
    for i in tqdm(range(len(name_list))):
        img_path = os.path.join(root, name_list[i])
        img = cv2.imread(img_path)
        # try:
        objs = detect(model, img, 0.4, resize=True)[0]
        objss[name_list[i]] = objs
        # except:
        #     print (img_path)
        #     continue
    print ('predict done!')
    return objss


def make_file(objss, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, 'a') as f:
        for k, objs in objss.items():
            if objs==[]:
                continue
            f.write('# ' + k + '\n')
            for obj in objs:

                f.write(f'{obj.x} {obj.y} {obj.width} {obj.height} ')
                for lx, ly in obj.landmark:
                    f.write(f'{lx} {ly} 0.0 ')
                f.write('1.0\n')
    print ('anno done!')
    return

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_path = './model/model_file/dbface_big.pth'
    root = '/home/data/Datasets/WIDERFace/WIDER_val/20cut_img/images'
    label_path = '/home/data/Datasets/WIDERFace/WIDER_val/20cut_img/lc_list.txt'
    output_file = '/home/data/Datasets/WIDERFace/WIDER_val/label.txt'

    dbface = DBFace()
    dbface.eval()
    dbface.cuda()
    dbface.load(model_path, device={'cuda:7':'cuda:0'})

    name_list = get_root(label_path)
    objss = predict(root, name_list, dbface)
    make_file(objss, output_file)


