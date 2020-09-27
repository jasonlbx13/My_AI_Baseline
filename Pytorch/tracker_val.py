import numpy as np
import time
from tqdm import tqdm
import cv2
import os
import pandas as pd
from tracker_light import Tracker


def load_retina_file(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()

    name_list = []
    lc_list = []
    w_list, h_list = [], []
    landmark_list = []
    for line in data:
        line = line.strip('\n')
        if line.startswith('#'):
            name = line[2:]
        else:
            line = line.split(' ')
            name_list.append(name)
            lc_list.append([int(float(line[i])) for i in range(2)])
            w_list.append(int(float(line[2])))
            h_list.append(int(float(line[3])))
            landmark = [[0, 0] for i in range(5)]
            for i in range(5):
                x, y = line[i * 3 + 4:i * 3 + 6]
                landmark[i][0], landmark[i][1] = float(x), float(y)
            landmark_list.append(landmark)
    retina_df = pd.DataFrame(data={'name': name_list,
                                   'left_coor': lc_list,
                                   'w': w_list,
                                   'h': h_list,
                                   'landmarks': landmark_list})
    return retina_df

if __name__ == '__main__':

    ONet_file = './model/model_file/onnx_mnn/ONet-tracker-light-5_2.mnn'
    label_file = '/home/data/Datasets/track/track_val.txt'
    df = load_retina_file(label_file)
    anno_boxes = []
    for i in range(len(df)):
        x1, y1 = df.iloc[i]['left_coor']
        x2 = x1 + df.iloc[i]['w']
        y2 = y1 + df.iloc[i]['h']
        box = [int(x1), int(y1), int(x2), int(y2)]
        anno_boxes.append(box)

    tk = Tracker(ONet_file)
    boxes = []
    x1_loss = []
    y1_loss = []
    x2_loss = []
    y2_loss = []
    cut_tmp = None
    find_face = 1
    fc = True

    for i in tqdm(range(300)):
        img_name = df.iloc[i]['name']
        img = cv2.imread('/home/data/Datasets/track/track_val/{}'.format(img_name))
        if i % 5 == 0:
            boxes = [anno_boxes[i]]
            cut_tmp = None
        else:
            objs = []
            cut_img_list, cut_tmp = tk.cut(img, boxes, cut_tmp)
            boxes = []
            for j in range(len(cut_img_list)):
                cut_img = np.array(cut_img_list[j]['img'], np.uint8)  # 裁剪好的带输入图像
                cut_img = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
                x1, y1 = cut_img_list[j]['lc']  # 裁剪区域左上角坐标
                old_box = cut_img_list[j]['ob']  # 上一帧的人脸框预测位置
                obj = tk.track_predict(cut_img, 0.5)  # 跟踪器预测当前帧人脸位置
                if obj == {}:
                    continue
                # 根据裁剪区域左上角坐标将输出转换为原图的绝对坐标
                obj['box'][0] += x1
                obj['box'][2] += x1
                obj['box'][1] += y1
                obj['box'][3] += y1

                if fc:
                    for k in range(4):  # 根据上一帧的预测框去抖，如果某条边变化不超过1个像素则不更新
                        if abs(obj['box'][k] - old_box[k]) < 1:
                            obj['box'][k] = old_box[k]

                objs.append(obj)
            if objs == []:
                print (i)
                objs = [{'box': old_box, 'score': 1}]
            for obj in objs:
                boxes.append(obj['box'])

            w = anno_boxes[i][2] - anno_boxes[i][0]
            h = anno_boxes[i][3] - anno_boxes[i][1]
            loss = (np.array(anno_boxes[i]) - np.array(boxes[0])) / (np.array([w, h, w, h]))
            x1_loss.append(abs(loss[0]))
            y1_loss.append(abs(loss[1]))
            x2_loss.append(abs(loss[2]))
            y2_loss.append(abs(loss[3]))

        # for l in range(len(boxes)):
        #     cv2.rectangle(img, (int(boxes[l][0]), int(boxes[l][1])), (int(boxes[l][2]), int(boxes[l][3])),
        #                   (0, 255, 0), 2)
        # cv2.imshow('im', img)  # 展示第二帧的图像和追踪到的人脸
        # cv2.waitKey(0)

    x1_loss = np.array(x1_loss).mean()
    y1_loss = np.array(y1_loss).mean()
    x2_loss = np.array(x2_loss).mean()
    y2_loss = np.array(y2_loss).mean()
    mean_loss = (x1_loss+y1_loss+x2_loss+y2_loss) / 4

    print ('############### LOSS ###############')
    print (f'x1 loss: {x1_loss}')
    print (f'y1 loss: {y1_loss}')
    print (f'x2 loss: {x2_loss}')
    print (f'y2 loss: {y2_loss}')
    print (f'mean loss: {mean_loss}')
    print ('####################################')







