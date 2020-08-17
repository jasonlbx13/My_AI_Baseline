import os
import numpy as np
import time
import MNN
import cv2
import heapq
import math

class Tracker:

    def __init__(self, tracker):
        '''
        初始化函数
        :param tracker: 跟踪器模型文件路径
        '''
        self.interpreter1 = MNN.Interpreter(tracker)
        self.session1 = self.interpreter1.createSession()
        self.input_tensor1 = self.interpreter1.getSessionInput(self.session1, 'input_data')

    def track_predict(self, img, threshold=0.4):
        '''
        跟踪器预测函数
        :param img: opencv格式图片，(H,W,C)
        :param threshold:  预测阈值，小于该阈值则判定为非人脸不显示预测框
        :return: 含有结果的字典 obj{'box':[a,b,c,d], score: 0.9}
        '''
        h, w, c = img.shape
        box = [0, 0, w, h]
        obj = {}

        if w != h:
            box = self.convert_to_square(box)
        else:
            pass
        box[0:4] = np.round(box[0:4])
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self.pad(box, w, h)
        cropped_ims = np.zeros((1, 48, 48, 3), dtype=np.float32)

        tmp = np.zeros((tmph, tmpw, 3), dtype=np.uint8)
        tmp[dy:edy + 1, dx:edx + 1, :] = img[y:ey + 1, x:ex + 1, :]
        cropped_ims[0, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128
        input = cropped_ims.transpose(0, 3, 1, 2)
        tmp_input = MNN.Tensor((1, 3, 48, 48), MNN.Halide_Type_Float,
                               input, MNN.Tensor_DimensionType_Caffe)
        self.input_tensor1.copyFrom(tmp_input)
        self.interpreter1.runSession(self.session1)

        cls_prob = self.interpreter1.getSessionOutput(self.session1, 'y_cls_prob').getData()
        bbox_pred = self.interpreter1.getSessionOutput(self.session1, 'y_bbox_pred').getData()
        if cls_prob[1] > threshold:
            box_c = self.calibrate_box(box, bbox_pred)
            corpbbox = [box_c[0], box_c[1], box_c[2], box_c[3]]
            obj['box'] = corpbbox
            obj['score'] = cls_prob[1]

        return obj

    def convert_to_square(self, box):
        '''
        将输入至跟踪器的图片补全为正方形
        :param box: 输入图片的尺寸坐标 [x1, y1, x2, y2]
        :return: 调整后的正方形box，[x1, y1, x2, y2]
        '''

        square_box = box.copy()
        h = box[3] - box[1] + 1
        w = box[2] - box[0] + 1
        # 找寻正方形最大边长
        max_side = np.maximum(w, h)

        square_box[0] = box[0] + w * 0.5 - max_side * 0.5
        square_box[1] = box[1] + h * 0.5 - max_side * 0.5
        square_box[2] = square_box[0] + max_side - 1
        square_box[3] = square_box[1] + max_side - 1

        return square_box

    def pad(self, bboxes, w, h):
        '''
        将超出图像的box进行处理
        :parambboxes:人脸框
        :paramw,h:图像长宽
        :return: 包含调整前后box位置信息的列表
                 dy, dx : 为调整后的box的左上角坐标相对于原box左上角的坐标
                 edy, edx : n为调整后的box右下角相对原box左上角的相对坐标
                 y, x : 调整后的box在原图上左上角的坐标
                 ex, ex : 调整后的box在原图上右下角的坐标
                 tmph, tmpw: 原始box的长宽
        '''
        #box的长宽
        tmpw, tmph = bboxes[2] - bboxes[0] + 1, bboxes[3] - bboxes[1] + 1
        num_box = 1

        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1
        #box左上右下的坐标
        x, y, ex, ey = bboxes[0], bboxes[1], bboxes[2], bboxes[3]
        #找到超出右下边界的box并将ex,ey归为图像的w,h
        #edx,edy为调整后的box右下角相对原box左上角的相对坐标
        if ex > w - 1:
            edx = tmpw + w - 2 - ex
            ex = w - 1

        if ey > h - 1:
            edy = tmph + h - 2 - ey
            ey = h - 1
        #找到超出左上角的box并将x,y归为0
        #dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
        if x < 0:
            dx = 0 - x
            x = 0

        if y < 0:
            dy = 0 - y
            y = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = np.array(return_list, dtype=np.int32)

        return return_list

    def calibrate_box(self, bbox, reg):
        '''
        生成预测的人脸检测框
        :param bbox: 输入至ONET追踪器的裁剪区域坐标
        :param reg: 追踪器预测的关于输入坐标的相对偏移量
        :return: 对应原图的人脸检测框坐标 [x1,y1,x2,y2]
        '''
        bbox_c = bbox.copy()
        w = bbox[2] - bbox[0] + 1
        w = np.expand_dims(w, 0)
        h = bbox[3] - bbox[1] + 1
        h = np.expand_dims(h, 0)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[0:4] = bbox_c[0:4] + aug
        return bbox_c

    def cut(self, img, boxes, old_cboxes):
        '''
        根据上一帧的先验信息，对当前帧图像进行适当裁剪以送入追踪器进行检测
        :param img: opencv格式图片，(H,W,C)
        :param boxes: 上一帧的预测框列表 [[x1,y1,x2,y2], ...]
        :param old_cboxes: 上一帧的裁剪区域列表 [[x1,y1,x2,y2], ...]
        :return: 当前帧的裁剪信息 {'img': crop_img, 'lc': left_coor, 'ob': [x1,y1,x2,y2]}
                 crop_img: 裁剪好的图片
                 lc: 裁剪区域的左上角坐标
                 ob: 对应当前人脸上一帧的裁剪区域(平滑用)
                 当前帧裁剪区域备份 [[x1,y1,x2,y2], ...]
        '''

        if boxes == []:
            return [], None
        height, width, _ = img.shape
        cut_img_list = []
        pad = 0.075
        new_boxes = []
        damping = 0.5  # 阻尼区占pad的比例
        for i in range(len(boxes)):

            x1, y1, x2, y2 = boxes[i]  # 上一帧预测的bbox
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            px = pad*w
            py = pad*h

            if old_cboxes is None:
                pass
            else:
                old_cbox = old_cboxes[i]
                old_w = old_cbox[2] - old_cbox[0]
                old_h = old_cbox[3] - old_cbox[1]
                ratio = 1/((1+2*pad)**2)
                # 贴边界和出边界的裁剪情况不启用阻尼和惯性机制，框面积缩小需要重新pad，故也不启用
                if old_cbox[0]<0 or old_cbox[1]<0 or old_cbox[2]>width or old_cbox[3]>height or (w*h)/(old_w*old_h)<(ratio-0.05):
                    pass
                else:
                    # 判断上一帧预测的bbox是否在惯性区内
                    if x1-damping*px>old_cbox[0] and y1-damping*py>old_cbox[1] and x2+damping*px<old_cbox[2] and y2+damping*py<old_cbox[3]:
                        crop_img = img[int(round(old_cbox[1])):int(round(old_cbox[3]))+1,
                                       int(round(old_cbox[0])):int(round(old_cbox[2]))+1, :]
                        new_boxes.append(old_cbox)
                        cut_img_list.append({'img': crop_img, 'lc': [old_cbox[0],old_cbox[1]], 'ob': [x1, y1, x2, y2]})
                        continue
                    # 若不在惯性区内，则判断上一帧预测的bbox是否有某条边进入阻尼区且没出边界
                    elif x1>old_cbox[0] and y1>old_cbox[1] and x2<old_cbox[2] and y2<old_cbox[3]:

                        if x1-damping*px<=old_cbox[0] and x2+damping*px>=old_cbox[2]: # 同时进入左右阻尼
                            old_cbox[0] = max(0, x1-damping*px)
                            old_cbox[2] = min(width, x2+damping*px)
                        elif x1-damping*px<=old_cbox[0]: # 进入了左阻尼区，没有进入右阻尼区
                            old_cbox[0] = max(0, x1-damping*px)
                            old_cbox[2] = old_cbox[0]+old_w
                        elif x2+damping*px>=old_cbox[2]: # 进入了右阻尼区，没有进入左阻尼区
                            old_cbox[2] = min(width, x2+damping*px)
                            old_cbox[0] = old_cbox[2]-old_w
                        else:
                            pass

                        if y1-damping*py<=old_cbox[1] and y2+damping*py>=old_cbox[3]: # 同时进入上下阻尼区
                            old_cbox[1] = max(0, y1-damping*py)
                            old_cbox[3] = min(height, y2+damping*py)
                        elif y1-damping*py<=old_cbox[1]: # 进入了上阻尼区，没有进入下阻尼区
                            old_cbox[1] = max(0, y1-damping*py)
                            old_cbox[3] = old_cbox[1]+old_h
                        elif y2+damping*py>=old_cbox[3]: # 进入了下阻尼区，没有进入上阻尼区
                            old_cbox[3] = min(height, y2+damping*py)
                            old_cbox[1] = old_cbox[3]-old_h
                        else:
                            pass
                        # 用微调后的旧裁剪边框剪切模型
                        crop_img = img[int(round(old_cbox[1])):int(round(old_cbox[3]))+1,
                                       int(round(old_cbox[0])):int(round(old_cbox[2]))+1, :]
                        new_boxes.append(old_cbox)
                        cut_img_list.append({'img': crop_img, 'lc': [old_cbox[0],old_cbox[1]], 'ob': [x1, y1, x2, y2]})
                        continue
                    else:
                        pass

            cx1 = max(0, int(x1-px))
            cy1 = max(0, int(y1-py))
            cx2 = min(width, int(x2+px))
            cy2 = min(height, int(y2+py))
            left_coor = [x1-px, y1-py]
            crop_img = img[cy1:cy2+1, cx1:cx2+1, :]
            crop_h,crop_w,_ = crop_img.shape
            # 若超出边界分八种情况处理，在缺失脸部分补黑边
            black_pad = np.zeros((int(round(h*(1+2*pad)))+1, int(round(w*(1+2*pad)))+1, 3))
            if x1-px<0 and y1-py>=0 and y2+py<=height: # 超左边界
                black_pad[:crop_h,int(px-x1):int(px-x1+crop_w),:] = crop_img
                crop_img = black_pad
            elif x1-px<0 and y1-py<0: # 超左上角
                black_pad[int(py-y1):int(py-y1+crop_h),int(px-x1):int(px-x1+crop_w),:] = crop_img
                crop_img = black_pad
            elif x1-px<0 and y2+py>height: # 超左下角
                black_pad[:crop_h,int(px-x1):int(px-x1+crop_w),:] = crop_img
                crop_img = black_pad
            elif x2+px>width and y1-py>=0 and y2+py<=height: # 超右边界
                black_pad[:crop_h,:crop_w,:] = crop_img
                crop_img = black_pad
            elif x2+px>width and y1-py<0: # 超右上角
                black_pad[int(py-y1):int(py-y1+crop_h),:crop_w,:] = crop_img
                crop_img = black_pad
            elif x2+px>width and y2+py>height: # 超右下角
                black_pad[:crop_h,:crop_w,:] = crop_img
                crop_img = black_pad
            elif y1-py<0 and x1-px>=0 and x2+px<=width: # 超上边界
                black_pad[int(py-y1):int(py-y1+crop_h),:crop_w,:] = crop_img
                crop_img = black_pad
            elif y2+py>height and x1-px>=0 and x2+px<=width: # 超下边界
                black_pad[:crop_h,:crop_w,:] = crop_img
                crop_img = black_pad
            else:
                pass
            cut_img_list.append({'img': crop_img, 'lc': left_coor, 'ob': [x1,y1,x2,y2]})
            new_boxes.append([x1-px, y1-py, x2+px, y2+py])
        return cut_img_list, new_boxes


if __name__ == '__main__':

    ONet_file = './model/model_file/onnx_mnn/ONet-tracker-7.mnn'
    tk = Tracker(ONet_file)

    # 逐帧跟踪
    # 第一帧检测帧
    img1 = cv2.imread('/home/data/Datasets/track/girl/imgs/img00000.png')
    boxes = [[145, 89, 222, 168]] # 模拟检测器检测出的人脸检测框
    for i in range(len(boxes)):
        cv2.rectangle(img1, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 0), 2)
    cv2.imshow('im', img1)  # 展示第一帧的图像和检测到的人脸
    cv2.waitKey(0)

    # 第二帧跟踪帧，根据上一帧检测器的结果进行跟踪
    img2 = cv2.imread('/home/data/Datasets/track/girl/imgs/img00001.png')
    objs = []
    cut_tmp = None
    cut_img_list, cut_tmp = tk.cut(img2, boxes, cut_tmp)
    boxes = []
    for i in range(len(cut_img_list)):
        cut_img = cut_img_list[i]['img'] # 裁剪好的带输入图像
        x1, y1 = cut_img_list[i]['lc'] # 裁剪区域左上角坐标
        old_box = cut_img_list[i]['ob'] # 上一帧的人脸框预测未知
        obj = tk.track_predict(cut_img) # 跟踪器预测当前帧人脸位置
        if obj == {}:
            continue
        # 根据裁剪区域左上角坐标将输出转换为原图的绝对坐标
        obj['box'][0] += x1
        obj['box'][2] += x1
        obj['box'][1] += y1
        obj['box'][3] += y1


        for i in range(4):  # 根据上一帧的预测框去抖，如果某条边变化不超过1个像素则不更新
            if abs(obj['box'][i] - old_box[i]) < 1:
                obj['box'][i] = old_box[i]
        objs.append(obj)
        boxes.append(obj['box'])

    for i in range(len(boxes)):
        cv2.rectangle(img2, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0), 2)
    cv2.imshow('im', img2)  # 展示第二帧的图像和追踪到的人脸
    cv2.waitKey(0)
    cv2.destroyAllWindows()

