import cv2
import time
import numpy as np
import onnx
import onnxruntime
import heapq
import MNN


class Infer:
    def __init__(self, model_file):

        self.model_type = model_file.split('.')[-1]
        if self.model_type=='onnx':
            self.onnx_model = onnx.load(model_file)
            onnx.checker.check_model(self.onnx_model)
            self.ort_session = onnxruntime.InferenceSession(onnx_file)
        else:
            self.interpreter = MNN.Interpreter(model_file)
            self.session = self.interpreter.createSession()
            self.input_tensor = self.interpreter.getSessionInput(self.session, 'input.1')
        self.mean = [0.408, 0.447, 0.47]
        self.std = [0.289, 0.274, 0.278]

    def predict(self, image, threshold=0.4):
        a = time.time()
        w, h = image.shape[1], image.shape[0]
        max_wh = max(w, h)
        scale = max_wh / 256
        newImage = np.zeros((max_wh, max_wh, 3), np.uint8)
        newImage[:h, :w, :] = image
        image = cv2.resize(newImage, (256, 256))
        image = ((image / 255. - self.mean) / self.std).astype(np.float32)
        input = image.transpose(2, 0, 1)[np.newaxis, :]
        print('qianchuli:{}'.format(time.time() - a))
        if self.model_type == 'onnx':
            ort_inputs = {self.ort_session.get_inputs()[0].name: input}
            net_start = time.time()
            hm, box = self.ort_session.run([self.ort_session.get_outputs()[i].name for i in range(2)], ort_inputs)
            print ('Net Time: {}'.format(time.time()-net_start))
            tb = time.time()

        else:
            c = time.time()
            tmp_input = MNN.Tensor((1, 3, 256, 256), MNN.Halide_Type_Float, \
                                   input, MNN.Tensor_DimensionType_Caffe)
            self.input_tensor.copyFrom(tmp_input)
            print('mnn_qianchuli:{}'.format(time.time() - c))
            net_start = time.time()
            self.interpreter.runSession(self.session)
            print('Net Time: {}'.format(time.time() - net_start))
            tb = time.time()
            hm = self.interpreter.getSessionOutput(self.session, '641')
            box = self.interpreter.getSessionOutput(self.session, '642')

            # hm = np.array(hm.getData()).reshape((64, 64, 4))[:, :, 0]
            # box = np.array(box.getData()).reshape((1, 64, 64, 4)).transpose(0, 3, 1, 2)

            tmp_hm = MNN.Tensor((1, 1, 64, 64), MNN.Halide_Type_Float, \
                                   hm.getData(), MNN.Tensor_DimensionType_Caffe)
            hm.copyToHostTensor(tmp_hm)
            hm = tmp_hm.getData()
            tmp_box = MNN.Tensor((1, 4, 64, 64), MNN.Halide_Type_Float, \
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

        print('houchuli:{}'.format(time.time()-tb))
        return self.nms(objs)

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


    def to_numpy(self, tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.cpu().numpy()

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

    def camera(self, threshold = 0.4):

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
        ok, frame = cap.read()

        pad = True

        while ok:
            if pad:
                w, h = frame.shape[1], frame.shape[0]
                max_wh = max(w, h)
                new_frame = np.zeros((max_wh, max_wh, 3), np.uint8)
                new_frame[:h, :w, :] = frame
                frame = new_frame
                frame = cv2.resize(frame, (256, 256))

            frame = cv2.flip(frame, 1)
            # frame = cv2.rotate(frame, cv2.ROTATE_180)
            objs = self.predict(frame, threshold)

            self.draw(frame, objs, camera=True)

            cv2.imshow("DBFace small", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            ok, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    onnx_file = './model/model_file/onnx_mnn/dbface_light_nolandmark.onnx'
    mnn_file = './model/model_file/onnx_mnn/dbface_light4.mnn'
    # mnn_file = './model/model_file/onnx_mnn/quan.mnn'
    infer = Infer(mnn_file)
    infer.camera(0.72)

    # img_path = '/home/data/TestImg/zipai/zipai5.jpg'
    # image = cv2.imread(img_path)
    # # for i in range(100):
    # # start = time.time()
    # objs = infer.predict(image, threshold=0.7)
    # # print ('Inference Time:{}'.format(time.time() - start))
    # infer.draw(image, objs)


