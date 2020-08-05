from utils import preprocess
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import os
# from model.dbface_small import DBFace
from model.dbface_light import DBFace
import time



def nms(objs, iou=0.5):

    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def detect(model, image, threshold=0.4, nms_iou=0.5, resize=True):

    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    if resize:
        size = 256
        w, h = image.shape[1], image.shape[0]
        max_wh = max(w, h)
        scale = max_wh / size
        newImage = np.zeros((max_wh, max_wh, 3), np.uint8)
        newImage[:h, :w, :] = image
        image = cv2.resize(newImage, (size, size))
    else:
        image = preprocess.pad(image)

    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]
    if torch.cuda.is_available():
        torch_image = torch_image.cuda()
    a = time.time()
    hm, box, landmark = model(torch_image)
    # print(time.time()-a)
    time_line = time.time()-a
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    # scores, indices = hm.view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]
    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    cnt = 0
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break
        cnt += 1

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (preprocess.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        if resize:
            xyrb *= np.array([scale] * 4)
            x5y5 *= np.array([scale] * 10)
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(preprocess.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    # print ('Face Num:{}'.format(cnt))
    return nms(objs, iou=nms_iou), time_line
    # return objs

def detect_image(model_path, img_path, threshold=0.66):

    dbface = DBFace(has_landmark=True, wide=64, has_ext=True, upmode="UCBA")
    dbface.eval()
    if torch.cuda.is_available():
        dbface.cuda()
    dbface.load(model_path)

    img = cv2.imread(img_path)
    H, W, C = img.shape
    objs, t = detect(dbface, img, threshold)
    for obj in objs:
        print ((obj.width*obj.height)/(H*W))
        preprocess.drawbbox(img, obj)

    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return t

def camera(model_path, threshold=0.4):

    dbface = DBFace(has_landmark=True, wide=24, has_ext=False, upmode="UCBA", compress=0.5)
    dbface.eval()
    dbface.load(model_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)  # 设置宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)  # 设置长度
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, frame = cap.read()

    while ok:

        frame = cv2.flip(frame, 1)
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        objs, t = detect(dbface, frame, threshold)

        for obj in objs:
            preprocess.drawbbox(frame, obj)

        cv2.imshow("demo DBFace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        ok, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model_path = './model/model_file/143.pth'
    # img_path = "/home/data/Datasets/SD/self_test/images/00000.jpg"

    # time_line = 0
    # for i in range(15):
    #     img_path = f"/home/data/TestImg/zipai/zipai{i}.jpg"
    #     time_line += detect_image(model_path, img_path, 0.4)
    # print (time_line/100)
    camera(model_path, 0.75)