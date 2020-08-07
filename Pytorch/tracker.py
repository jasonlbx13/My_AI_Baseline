import os
import numpy as np
import time
import MNN
import cv2
import heapq
import math

class Tracker:

    def __init__(self, tracker, corrector):
        self.interpreter1 = MNN.Interpreter(tracker)
        self.session1 = self.interpreter1.createSession()
        self.input_tensor1 = self.interpreter1.getSessionInput(self.session1, 'input_data')

        self.interpreter2 = MNN.Interpreter(corrector)
        self.session2 = self.interpreter2.createSession()
        self.input_tensor2 = self.interpreter2.getSessionInput(self.session2, 'input.1')

        self.mean = [0.408, 0.447, 0.47]
        self.std = [0.289, 0.274, 0.278]

    def track_predict(self, img, threshold=0.4):

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
        input = cropped_ims.transpose(0,3,1,2)
        tmp_input = MNN.Tensor((1, 3, 48, 48), MNN.Halide_Type_Float,
                               input, MNN.Tensor_DimensionType_Caffe)
        self.input_tensor1.copyFrom(tmp_input)
        a = time.time()
        self.interpreter1.runSession(self.session1)
        print(time.time() - a)
        cls_prob = self.interpreter1.getSessionOutput(self.session1, 'y_cls_prob').getData()
        bbox_pred = self.interpreter1.getSessionOutput(self.session1, 'y_bbox_pred').getData()
        if cls_prob[1] > threshold:
            box_c = self.calibrate_box(box, bbox_pred)
            corpbbox = [int(box_c[0]), int(box_c[1]), int(box_c[2]), int(box_c[3])]
            obj['box'] = corpbbox
            obj['score'] = cls_prob[1]

        return obj

    def correct_predict(self, img, threshold=0.4):

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

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

        face_num = 0
        boxes = []
        flag = 0

        while 1:
            tmp = []
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = cv2.flip(frame, 1)
            # frame = cv2.rotate(frame, cv2.ROTATE_180)

            if flag == 0:
                objs = self.correct_predict(frame, threshold_correct)

                face_num = len(objs)
                if objs == []:
                    boxes = []
                else:
                    for obj in objs:
                        tmp.append(list(map(int,obj['box'])))
                    boxes = tmp
                    flag = 1
            else:
                if boxes == []:
                    objs = []
                else:
                    cut_img_list = self.cut(frame, boxes)
                    objs = []
                    for i in range(face_num):
                        cut_img = cut_img_list[i]['img']
                        x1, y1 = cut_img_list[i]['lc']
                        old_box = cut_img_list[i]['ob']
                        obj = self.track_predict(cut_img, threshold_track)
                        if obj == {}:
                            continue
                        obj['box'][0] += x1
                        obj['box'][2] += x1
                        obj['box'][1] += y1
                        obj['box'][3] += y1
                        beta = 0
                        obj['box'] = list(beta*np.array(old_box)+(1-beta)*np.array(obj['box']))
                        objs.append(obj)
                        tmp.append(obj['box'])
                    boxes = tmp
            if objs!=[]:
                self.draw(frame, objs, camera=True)
            else:
                flag = 0

            cv2.imshow("tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def draw(self, image, objs, camera=False):

        for obj in objs:
            x, y, r, b = [int(obj['box'][i]) for i in range(4)]
            cv2.rectangle(image, (x, y, r - x + 1, b - y + 1), (0, 255, 0), 2, 16)
            text = f"{obj['score']:.2f}"
            cv2.putText(image, text, (x + 3, y - 5), 0, 0.5, (0, 0, 0), 1, 16)
            # for i in range(len(obj['landmark'])):
            #     x, y = obj['landmark'][i][:2]
            #     cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1, 16)
        if not camera:
            cv2.imshow('result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            pass

    def nms(self, objs, iou=0.5):
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
        '''将box转换成更大的正方形
        参数：
          box：预测的box,[n,5]
        返回值：
          调整后的正方形box，[n,5]
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
        '''将超出图像的box进行处理
        参数：
          bboxes:人脸框
          w,h:图像长宽
        返回值：
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

        bbox_c = bbox.copy()
        w = bbox[2] - bbox[0] + 1
        w = np.expand_dims(w, 0)
        h = bbox[3] - bbox[1] + 1
        h = np.expand_dims(h, 0)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[0:4] = bbox_c[0:4] + aug
        return bbox_c

    def cut(self, img, boxes):

        if boxes == []:
            return []
        height, width, _ = img.shape
        cut_img_list = []
        pad = 0.1
        for box in boxes:

            x1, y1, x2, y2 = box

            w = int(x2 - x1 + 1)
            h = int(y2 - y1 + 1)
            px = pad*w
            py = pad*h

            cx1 = max(0, int(x1-px))
            cy1 = max(0, int(y1-py))
            cx2 = min(width, int(x2+px))
            cy2 = min(height, int(y2+py))
            left_coor = [int(x1-px), int(y1-py)]
            crop_img = img[cy1:cy2+1, cx1:cx2+1, :]
            crop_h,crop_w,_ = crop_img.shape

            black_pad = np.zeros((int(h*(1+2*pad))+1, int(w*(1+2*pad))+1, 3))
            if x1-px<0 and y1-py>=0 and y2+py<=height:
                black_pad[:crop_h,int(px-x1):int(px-x1+crop_w),:] = crop_img
                crop_img = black_pad
            elif x1-px<0 and y1-py<0:
                black_pad[int(py-y1):int(py-y1+crop_h),int(px-x1):int(px-x1+crop_w),:] = crop_img
                crop_img = black_pad
            elif x1-px<0 and y2+py>height:
                black_pad[:crop_h,int(px-x1):int(px-x1+crop_w),:] = crop_img
                crop_img = black_pad
            elif x2+px>width and y1-py>=0 and y2+py<=height:
                black_pad[:crop_h,:crop_w,:] = crop_img
                crop_img = black_pad
            elif x2+px>width and y1-py<0:
                black_pad[int(py-y1):int(py-y1+crop_h),:crop_w,:] = crop_img
                crop_img = black_pad
            elif x2+px>width and y2+py>height:
                black_pad[:crop_h,:crop_w,:] = crop_img
                crop_img = black_pad
            elif y1-py<0 and x1-px>=0 and x2+px<=width:
                black_pad[int(py-y1):int(py-y1+cy2-cy1+1),:crop_w,:] = crop_img
                crop_img = black_pad
            elif y2+py>height and x1-px>=0 and x2+px<=width:
                black_pad[:crop_h,:crop_w,:] = crop_img
                crop_img = black_pad
            else:
                pass

            cut_img_list.append({'img': crop_img, 'lc': left_coor, 'ob': [x1,y1,x2,y2]})

        return cut_img_list


if __name__ == '__main__':

    ONet_file = './model/model_file/onnx_mnn/ONet-tracker-2.mnn'
    DBFace_file = './model/model_file/onnx_mnn/dbface_light4.mnn'
    image_file = '/home/data/TestImg/pad1.png'
    tk = Tracker(ONet_file, DBFace_file)

    tk.track(0.72,0.2)


    # boxes = []
    # img = cv2.imread(image_file)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # objs = tk.correct_predict(img)
    #
    # for obj in objs:
    #     boxes.append(obj['box'])
    # face_num = len(objs)
    # cut_img_list = tk.cut(img, boxes)
    # objs = []
    # for i in range(face_num):
    #     cut_img = cut_img_list[i]['img']
    #     cv2.imshow('cut_img', cut_img)
    #     cv2.waitKey(0)
    #     x1, y1 = cut_img_list[i]['lc']
    #     obj = tk.track_predict(cut_img, 0.4)
    #     if obj == {}:
    #         continue
    #     obj['box'][0] += x1
    #     obj['box'][2] += x1
    #     obj['box'][1] += y1
    #     obj['box'][3] += y1
    #     objs.append(obj)
    #     box = obj['box']
    #     cv2.rectangle(img, (int(box[0]), int(box[1])),
    #                   (int(box[2]), int(box[3])), (255, 0, 0), 1)
    # cv2.imshow('pred_img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
