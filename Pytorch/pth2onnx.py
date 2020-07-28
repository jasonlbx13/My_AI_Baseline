import os
from model.dbface_small import DBFace
import cv2
import torch
import numpy as np


def onnx(model, output_path):

    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]
    image = cv2.imread("/home/data/TestImg/tuchong/4.jpg")
    image = ((image/255.-mean)/std).astype(np.float32)
    image = cv2.resize(image, (256, 256))
    input = image.transpose(2, 0, 1)
    input = torch.from_numpy(input)[None]
    # Export the model
    torch.onnx.export(model,
                      input,
                      output_path,
                      verbose=True)

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model_path = "./model/model_file/dbface_nearsmall_rubust_selfdata5.pth"
    output_path = "./model/model_file/onnx_mnn/dbface_small_nolandmark.onnx"
    landmark = False
    has_ext = True
    upmode = 'UCBA'
    wide = 64
    if landmark:
        model = DBFace(has_landmark=True, wide=wide, has_ext=has_ext, upmode=upmode)
        model.load(model_path)
    else:
        model = DBFace(has_landmark=False, wide=wide, has_ext=has_ext, upmode=upmode)
        state_dict = torch.load(model_path, map_location='cpu')
        del_key = []
        for key in state_dict.keys():
            if key[0:8]=='landmark':
                del_key.append(key)
        for key in del_key:
            state_dict.pop(key)
        model.load_state_dict(state_dict)
    model.eval()
    onnx(model, output_path)
