import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import preprocess
import torch
import torch.nn as nn
from utils import logger
import numpy as np
from model.dbface_light import DBFace
from evaluate import evaluation


def detect_image(model, image, mean, std, threshold=0.4):
    image = preprocess.pad(image)
    image = ((image / 255 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0).cuda()
    center, box, landmark = model(image)
    stride = 4

    _, num_classes, hm_height, hm_width = center.shape
    hm = center[0].reshape(1, num_classes, hm_height, hm_width)
    tlrb = box[0].cpu().data.numpy().reshape(1, num_classes * 4, hm_height, hm_width)
    landmark = landmark[0].cpu().data.numpy().reshape(1, num_classes * 10, hm_height, hm_width)

    nmskey = _nms(hm, 3)
    kscore, kinds, kcls, kys, kxs = _topk(nmskey, 2000)
    kys = kys.cpu().data.numpy().astype(np.int)
    kxs = kxs.cpu().data.numpy().astype(np.int)
    kcls = kcls.cpu().data.numpy().astype(np.int)

    key = [[], [], [], []]
    for ind in range(kscore.shape[1]):
        score = kscore[0, ind]
        if score > threshold:
            key[0].append(kys[0, ind])
            key[1].append(kxs[0, ind])
            key[2].append(score)
            key[3].append(kcls[0, ind])

    imboxs = []
    if key[0] is not None and len(key[0]) > 0:
        ky, kx = key[0], key[1]
        classes = key[3]
        scores = key[2]

        for i in range(len(kx)):
            class_ = classes[i]
            cx, cy = kx[i], ky[i]
            x1, y1, x2, y2 = tlrb[0, class_ * 4:(class_ + 1) * 4, cy, cx]
            x1, y1, x2, y2 = (np.array([cx, cy, cx, cy]) + np.array([-x1, -y1, x2, y2])) * stride

            x5y5 = landmark[0, 0:10, cy, cx]
            x5y5 = np.array(preprocess.exp(x5y5 * 4))
            x5y5 = (x5y5 + np.array([cx] * 5 + [cy] * 5)) * stride
            boxlandmark = list(zip(x5y5[:5], x5y5[5:]))
            imboxs.append(preprocess.BBox(label=str(class_), xyrb=preprocess.floatv([x1, y1, x2, y2]), score=scores[i].item(),
                                      landmark=boxlandmark))
    return imboxs


def detect_images_giou_with_netout(output_hm, output_tlrb, output_landmark, threshold=0.4, ibatch=0):

    stride = 4
    _, num_classes, hm_height, hm_width = output_hm.shape
    hm = output_hm[ibatch].reshape(1, num_classes, hm_height, hm_width)
    tlrb = output_tlrb[ibatch].cpu().data.numpy().reshape(1, num_classes * 4, hm_height, hm_width)
    landmark = output_landmark[ibatch].cpu().data.numpy().reshape(1, num_classes * 10, hm_height, hm_width)

    nmskey = _nms(hm, 3)
    kscore, kinds, kcls, kys, kxs = _topk(nmskey, 2000)
    kys = kys.cpu().data.numpy().astype(np.int)
    kxs = kxs.cpu().data.numpy().astype(np.int)
    kcls = kcls.cpu().data.numpy().astype(np.int)

    key = [[], [], [], []]
    for ind in range(kscore.shape[1]):
        score = kscore[0, ind]
        if score > threshold:
            key[0].append(kys[0, ind])
            key[1].append(kxs[0, ind])
            key[2].append(score)
            key[3].append(kcls[0, ind])

    imboxs = []
    if key[0] is not None and len(key[0]) > 0:
        ky, kx = key[0], key[1]
        classes = key[3]
        scores = key[2]

        for i in range(len(kx)):
            class_ = classes[i]
            cx, cy = kx[i], ky[i]
            x1, y1, x2, y2 = tlrb[0, class_*4:(class_+1)*4, cy, cx]
            x1, y1, x2, y2 = (np.array([cx, cy, cx, cy]) + np.array([-x1, -y1, x2, y2])) * stride

            x5y5 = landmark[0, 0:10, cy, cx]
            x5y5 = np.array(preprocess.exp(x5y5 * 4))
            x5y5 = (x5y5 + np.array([cx]*5 + [cy]*5)) * stride
            boxlandmark = list(zip(x5y5[:5], x5y5[5:]))
            imboxs.append(preprocess.BBox(label=str(class_), xyrb=preprocess.floatv([x1, y1, x2, y2]), score=scores[i].item(), landmark=boxlandmark))
    return imboxs

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep
    # hmax = heat
    # return hmax


def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


if __name__ == '__main__':
    # create logger
    trial_name = "dbface_nearsmall_rubust_selfdata5"
    jobdir = './output/eval_result'
    model_path = './model/model_file/150.pth'
    log = logger.create(trial_name, f"{jobdir}/eval.log")

    # load and init model
    model = DBFace(has_landmark=True, wide=24, has_ext=False, upmode="UCBA")
    model.load(model_path)
    model.eval()
    model.cuda()

    # load dataset
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]
    files, anns = zip(*preprocess.load_webface("/home/data/Datasets/WIDERFace/retinaface_labels/val/label.txt",
                                           "/home/data/Datasets/WIDERFace/WIDER_val/images"))


    # forward and summary
    prefix = "/home/data/Datasets/WIDERFace/WIDER_val/images/"
    all_result_dict = {}
    total_file = len(files)

    for i in range(total_file):

        # preper key and file_name
        file = files[i]
        key = file[len(prefix): file.rfind("/")]
        file_name = preprocess.file_name_no_suffix(file)

        # load image and forward
        image = preprocess.imread(file)
        objs = detect_image(model, image, mean, std, 0.05)

        # summary to all_result_dict
        image_pred = []
        for obj in objs:
            image_pred.append(obj.xywh + [obj.score])

        # build all_result_dict
        if key not in all_result_dict:
            all_result_dict[key] = {}

        all_result_dict[key][file_name] = np.array(image_pred)
        log.info("{} / {}".format(i + 1, total_file))

        # write matlab format
        path = f"{jobdir}/result/{key}/{file_name}.txt"
        preprocess.mkdirs_from_file_path(path)

        with open(path, "w") as f:
            f.write(f"/{key}/{file_name}\n{len(image_pred)}\n")

            for item in image_pred:
                f.write("{} {} {} {} {}\n".format(*item))

    # eval map of IoU0.5
    aps = evaluation.eval_map(all_result_dict, all=False)

    log.info("\n"
             "Easy:      {}\n"
             "Medium:    {}\n"
             "Hard:      {}".format(*aps)
             )