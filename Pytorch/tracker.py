import os
import numpy as np
import time
import MNN
import cv2
import heapq
import math

class Tracker:

    def __init__(self, tracker, corrector):
        '''
        初始化函数
        :param tracker: 跟踪器模型文件路径
        :param corrector: 人脸检测器模型文件路径
        '''
        self.interpreter1 = MNN.Interpreter(tracker)
        self.session1 = self.interpreter1.createSession()
        self.input_tensor1 = self.interpreter1.getSessionInput(self.session1, 'input_data')

        self.interpreter2 = MNN.Interpreter(corrector)
        self.session2 = self.interpreter2.createSession()
        self.input_tensor2 = self.interpreter2.getSessionInput(self.session2, 'input.1')

        self.mean = [0.408, 0.447, 0.47]
        self.std = [0.289, 0.274, 0.278]

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
        a = time.time()
        self.interpreter1.runSession(self.session1)
        # print(time.time() - a)
        cls_prob = self.interpreter1.getSessionOutput(self.session1, 'y_cls_prob').getData()
        bbox_pred = self.interpreter1.getSessionOutput(self.session1, 'y_bbox_pred').getData()
        if cls_prob[1] > threshold:
            box_c = self.calibrate_box(box, bbox_pred)
            corpbbox = [box_c[0], box_c[1], box_c[2], box_c[3]]
            obj['box'] = corpbbox
            obj['score'] = cls_prob[1]

        return obj

    def correct_predict(self, img, threshold=0.72):
        '''
        人脸检测器预测函数
        :param img: opencv格式图片，(H,W,C)
        :param threshold:  预测阈值，小于该阈值则判定为非人脸不显示预测框
        :return: 人脸检测器预测的人脸集合 objs[obj{'box':[a,b,c,d], score: 0.9},...]
        '''
        w, h = img.shape[1], img.shape[0]
        max_wh = max(w, h)
        scale = max_wh / 256
        newImage = np.zeros((max_wh, max_wh, 3), np.uint8)
        newImage[:h, :w, :] = img
        image = cv2.resize(newImage, (256, 256))
        image = ((image / 255. - self.mean) / self.std).astype(np.float32)
        input = image.transpose(2, 0, 1)[np.newaxis, :]
        tmp_input = MNN.Tensor((1, 3, 256, 256), MNN.Halide_Type_Float,
                               input, MNN.Tensor_DimensionType_Caffe)
        self.input_tensor2.copyFrom(tmp_input)
        self.interpreter2.runSession(self.session2)
        hm = self.interpreter2.getSessionOutput(self.session2, '641')
        box = self.interpreter2.getSessionOutput(self.session2, '642')

        tmp_hm = MNN.Tensor((1, 1, 64, 64), MNN.Halide_Type_Float,
                            hm.getData(), MNN.Tensor_DimensionType_Caffe)
        hm.copyToHostTensor(tmp_hm)
        hm = tmp_hm.getData()
        tmp_box = MNN.Tensor((1, 4, 64, 64), MNN.Halide_Type_Float,
                             box.getData(), MNN.Tensor_DimensionType_Caffe)
        box.copyToHostTensor(tmp_box)
        box = tmp_box.getData()

        hm_line = hm.reshape((1, -1))[0]
        indices, scores = np.asarray(heapq.nlargest(100, enumerate(hm_line), lambda x: x[1]),
                                     dtype=np.float32).transpose()

        ys = indices / 64
        xs = indices % 64
        ys = [int(ys[i]) for i in range(len(ys))]
        xs = [int(xs[j]) for j in range(len(xs))]

        scores = list(scores)
        box = box[0]

        stride = 4
        objs = []

        for cx, cy, score in zip(xs, ys, scores):
            if score < threshold:
                break

            x, y, r, b = box[:, cy, cx]
            xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
            xyrb *= np.array([scale] * 4)
            objs.append({'box': xyrb, 'score': score})

        return self.nms(objs)

    def track(self, threshold_correct=0.5, threshold_track=0.5):
        '''
        跟踪函数，用于摄像头测试，实际部署时可省略
        :param threshold_correct: 人脸检测器阈值
        :param threshold_track:  人脸跟踪器阈值
        :return: 无返回值，直接开启摄像头进行测试
        '''
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

        face_num = 0
        boxes = []
        find_face = 0
        cut_tmp = None
        cnt = -1

        while 1:
            cnt+=1
            tmp = []
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = cv2.flip(frame, 1)
            # frame = cv2.rotate(frame, cv2.ROTATE_180)

            if find_face == 0:  # 如果是第一帧或上一帧tracker没有检测到人脸
                cnt = 0
                objs = self.correct_predict(frame, threshold_correct)
                face_num = len(objs)
                if objs == []:
                    boxes = []
                else:
                    for obj in objs:
                        tmp.append(obj['box'])
                    boxes = tmp
                    find_face = 1
            else:
                if boxes == []:
                    objs = []
                else:
                    # 根据上一帧的结果计算获取这一帧裁剪好的图片和和这一帧裁剪的box
                    cut_img_list, cut_tmp = self.cut(frame, boxes, cut_tmp)
                    objs = []
                    for i in range(len(cut_img_list)):
                        cut_img = cut_img_list[i]['img']
                        x1, y1 = cut_img_list[i]['lc']
                        old_box = cut_img_list[i]['ob']    # 上一帧检测出来的bbox
                        obj = self.track_predict(cut_img, threshold_track)
                        if obj == {}:
                            continue

                        obj['box'][0] += x1
                        obj['box'][2] += x1
                        obj['box'][1] += y1
                        obj['box'][3] += y1

                        # beta = 0
                        # obj['box'] = list(beta*np.array(old_box)+(1-beta)*np.array(obj['box']))  # 平滑

                        for i in range(4):  # 去抖
                            if abs(obj['box'][i]-old_box[i])<1:
                                obj['box'][i] = old_box[i]

                        objs.append(obj)
                        tmp.append(obj['box'])
                    boxes = tmp
                    face_num = len(objs)

                    if cnt % 5 == 0:  # 追踪若干帧后启用检测器进行校正判定
                        val_objs = self.correct_predict(frame, threshold_correct)
                        val_face_num = len(val_objs)
                        if val_face_num > face_num: # 如果检测器检测到的人脸多于跟踪器，则立即更新为检测器结果
                            tmp = []
                            objs = []
                            for obj in val_objs:
                                objs.append(obj)
                                tmp.append(obj['box'])
                            boxes = tmp
                            cut_tmp = None
                            face_num = val_face_num
                        elif val_face_num == face_num: # 若检测到的人脸数量相同，则通过IOU判断是否需要更新
                            for i in range(face_num):
                                max_iou = -1
                                max_val_box = []
                                for j in range(val_face_num):
                                    rec1 = objs[i]['box']
                                    rec2 = val_objs[j]['box']
                                    iou = tk.computeIOU(rec1, rec2)
                                    if iou > max_iou:
                                        max_val_box = val_objs[j]['box']
                                        max_iou = iou
                                if 0 < max_iou < 0.5:
                                    objs[i]['box'] = max_val_box
                                    cut_tmp = None
                                else:
                                    pass

                        else:
                            pass

            if objs!=[]: # 如果检测到人脸，则绘制人脸框
                self.draw(frame, objs, camera=True)
            else:
                find_face = 0
                cut_tmp = None

            cv2.imshow("tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def draw(self, image, objs, cut_box=None, camera=False):
        '''
        绘制标出人脸检测框后的图片，测试用， 部署时可忽略
        :param image: opencv格式图片，(H,W,C)
        :param objs: 人脸检测器或人脸跟踪器预测的人脸集合 objs[obj{'box':[a,b,c,d], score: 0.9},...]
        :param cut_box: 当前帧裁剪框的坐标 [x1, y1, x2, y2], 默认None为不绘制
        :param camera: 是否用于摄像头
        :return: 无返回，直接显示图片
        '''
        for j in range(len(objs)):
            x, y, r, b = [int(round(objs[j]['box'][i])) for i in range(4)]
            cv2.rectangle(image, (x, y, r - x + 1, b - y + 1), (0, 255, 0), 2, 16)
            if cut_box is not None:
                cx1, cy1, cx2, cy2 = [int(round(cut_box[j][i])) for i in range(4)]
                cv2.rectangle(image, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2, 16)
            text = f"{objs[j]['score']:.2f}"
            cv2.putText(image, text, (x + 3, y - 5), 0, 0.5, (0, 0, 0), 1, 16)

        if not camera:
            cv2.imshow('result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            pass

    def nms(self, objs, iou=0.5):
        '''
        NMS人脸框去重，用于滤掉多余的人脸检测框
        :param objs: 人脸检测器预测的人脸集合 objs[obj{'box':[a,b,c,d], score: 0.9},...]
        :param iou: iou阈值，大于该阈值的重复框会被舍弃
        :return: 返回过滤后的objs
        '''
        if objs is None or len(objs) <= 1:
            return objs

        objs = sorted(objs, key=lambda obj: obj['score'], reverse=True)
        keep = []
        flags = [0] * len(objs)
        for i, obj in enumerate(objs):

            if flags[i] != 0:
                continue

            keep.append(obj)
            for j in range(i + 1, len(objs)):
                if flags[j] == 0 and self.computeIOU(objs[i]['box'], objs[j]['box']) > iou:
                    flags[j] = 1
        return keep

    def computeIOU(self, rec1, rec2):
        '''
        就算两个检测狂IOU的函数
        :param rec1: 第一个检测框的坐标[x1,y1,x2,y2]
        :param rec2: 第二个检测框的坐标[x1,y1,x2,y2]
        :return: 返回两个检测框的IOU值
        '''
        cx1, cy1, cx2, cy2 = rec1
        gx1, gy1, gx2, gy2 = rec2
        S_rec1 = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
        S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
        x1 = max(cx1, gx1)
        y1 = max(cy1, gy1)
        x2 = min(cx2, gx2)
        y2 = min(cy2, gy2)

        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)
        area = w * h
        iou = area / (S_rec1 + S_rec2 - area)
        return iou


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
    DBFace_file = './model/model_file/onnx_mnn/dbface_light4.mnn'
    tk = Tracker(ONet_file, DBFace_file)

# 摄像头视频预测, 丢失人脸自动启动人脸检测器，其余时间除第一帧均为追踪器工作

    tk.track(0.72,0.5)

# 逐帧读入图片预测，除第一帧使用检测器外均为人脸追踪，丢失目标自动停止
#     track_images_root = '/home/data/Datasets/track/girl/imgs/'
#     boxes = []
#     img1 = cv2.imread(track_images_root+'img00001.png')
#     objs = tk.correct_predict(img1)
#     tk.draw(img1, objs, cut_box=None, camera=False)
#     face_nums = len(objs)
#     for obj in objs:
#         boxes.append(obj['box'])
#     cut_tmp = None
#
#     for i in range(2,100):
#         img2 = cv2.imread(track_images_root+'img{}.png'.format(str(i).zfill(5)))
#         objs = []
#         cut_img_list, cut_tmp = tk.cut(img2, boxes, cut_tmp)  # 读取这一帧裁剪好的图片和和这一帧裁剪的box
#         for i in range(face_nums):
#             cut_img = cut_img_list[i]['img'] # 裁剪后带输入的图片
#             x1, y1 = cut_img_list[i]['lc']  # 裁剪区域的左上角坐标
#             old_box = cut_img_list[i]['ob']  # 上一帧检测出来的bbox
#             obj = tk.track_predict(cut_img, 0.01)  # 第i个脸的预测结果
#             if obj == {}:
#                 continue
#             obj['box'][0] += x1
#             obj['box'][2] += x1
#             obj['box'][1] += y1
#             obj['box'][3] += y1
#             objs.append(obj)
#         face_nums = len(objs)  # 计算tracker找到的人脸数量
#         boxes = []
#         if objs == []:
#             print ('丢失人脸')
#             break
#         for obj in objs:
#             boxes.append(obj['box'])
#         tk.draw(img2, objs, cut_tmp, camera=False)

# 逐帧读入图片，检测器会固定若干帧进行一次检测，若跟踪器丢失人脸自动启用检测器

    # track_images_root = '/home/data/Datasets/track/girl/imgs/'
    # boxes = []
    # cnt = 0
    # face_nums = 0
    # cut_tmp = None
    #
    # for i in range(410, 490):
    #
    #     img = cv2.imread(track_images_root+'img{}.png'.format(str(i).zfill(5)))
    #     if face_nums != 0 and cnt % 5 != 0:
    #         objs = []
    #         cut_img_list, cut_tmp = tk.cut(img, boxes, cut_tmp)  # 读取这一帧裁剪好的图片和和这一帧裁剪的box
    #         for i in range(face_nums):
    #             cut_img = cut_img_list[i]['img'] # 裁剪后带输入的图片
    #             x1, y1 = cut_img_list[i]['lc']  # 裁剪区域的左上角坐标
    #             old_box = cut_img_list[i]['ob']  # 上一帧检测出来的bbox
    #             obj = tk.track_predict(cut_img, 0.5)  # 第i个脸的预测结果
    #             if obj == {}:
    #                 continue
    #             obj['box'][0] += x1
    #             obj['box'][2] += x1
    #             obj['box'][1] += y1
    #             obj['box'][3] += y1
    #             objs.append(obj)
    #
    #         if objs == []:
    #             face_nums = 0
    #
    #     if face_nums == 0 or cnt % 5 == 0:
    #         cut_tmp = None
    #         objs = tk.correct_predict(img,0.4)
    #         if objs == []:
    #             cnt = 0
    #             cv2.imshow('im', img)
    #             key = cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    #             if key == 27:
    #                 break
    #             continue
    #
    #     face_nums = len(objs)
    #     boxes = []
    #     for obj in objs:
    #         boxes.append(obj['box'])
    #     tk.draw(img, objs, cut_tmp, camera=False)
    #     cnt += 1

