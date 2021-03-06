from utils import preprocess, augment, logger
from model import losses
import numpy as np
import math
import random
from tqdm import tqdm
import torch
import torchvision.transforms.functional as T
from torch.utils.data import Dataset, DataLoader
from model.dbface_big import DBFace
from tensorboardX import SummaryWriter
import cv2
import eval


class LDataset(Dataset):
    def __init__(self, labelfile, imagesdir, mean, std, width=800, height=800):
        
        self.width = width
        self.height = height
        self.items = preprocess.load_webface(labelfile, imagesdir)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        imgfile, objs = self.items[index]
        image = preprocess.imread(imgfile)

        if image is None:
            log.info("{} is empty, index={}".format(imgfile, index))
            return self[random.randint(0, len(self.items)-1)]

        keepsize            = 10
        image, objs = augment.webface(image, objs, self.width, self.height, keepsize=0)

        # norm
        image = ((image / 255.0 - self.mean) / self.std).astype(np.float32)

        posweight_radius    = 2
        stride              = 4
        fm_width            = self.width // stride
        fm_height           = self.height // stride

        heatmap_gt          = np.zeros((1,     fm_height, fm_width), np.float32)
        heatmap_posweight   = np.zeros((1,     fm_height, fm_width), np.float32)
        keep_mask           = np.ones((1,     fm_height, fm_width), np.float32)
        reg_tlrb            = np.zeros((1 * 4, fm_height, fm_width), np.float32)
        reg_mask            = np.zeros((1,     fm_height, fm_width), np.float32)
        distance_map        = np.zeros((1,     fm_height, fm_width), np.float32) + 1000
        landmark_gt         = np.zeros((1 * 10,fm_height, fm_width), np.float32)
        landmark_mask       = np.zeros((1,     fm_height, fm_width), np.float32)

        for obj in objs:
            isSmallObj = obj.area < keepsize * keepsize
            if isSmallObj:
                cx, cy = obj.safe_scale_center(1 / stride, fm_width, fm_height)
                keep_mask[0, cy, cx] = 0
                w, h = obj.width / stride, obj.height / stride
                x0 = int(preprocess.clip_value(cx - w // 2, fm_width-1))
                y0 = int(preprocess.clip_value(cy - h // 2, fm_height-1))
                x1 = int(preprocess.clip_value(cx + w // 2, fm_width-1) + 1)
                y1 = int(preprocess.clip_value(cy + h // 2, fm_height-1) + 1)
                if x1 - x0 > 0 and y1 - y0 > 0:
                    keep_mask[0, y0:y1, x0:x1] = 0
            else:
                pass

        for obj in objs:
            if obj.neg_rotate:
                break
            classes = 0
            cx, cy = obj.safe_scale_center(1 / stride, fm_width, fm_height)
            reg_box = np.array(obj.box) / stride
            isSmallObj = obj.area < keepsize * keepsize

            if isSmallObj:
                if obj.area >= 5 * 5:
                    distance_map[classes, cy, cx] = 0
                    reg_tlrb[classes*4:(classes+1)*4, cy, cx] = reg_box
                    reg_mask[classes, cy, cx] = 1
                continue

            w, h = obj.width / stride, obj.height / stride
            x0 = int(preprocess.clip_value(cx - w // 2, fm_width-1))
            y0 = int(preprocess.clip_value(cy - h // 2, fm_height-1))
            x1 = int(preprocess.clip_value(cx + w // 2, fm_width-1) + 1)
            y1 = int(preprocess.clip_value(cy + h // 2, fm_height-1) + 1)
            if x1 - x0 > 0 and y1 - y0 > 0:
                keep_mask[0, y0:y1, x0:x1] = 1

            w_radius, h_radius = preprocess.truncate_radius((obj.width, obj.height))
            gaussian_map = preprocess.draw_truncate_gaussian(heatmap_gt[classes, :, :], (cx, cy), h_radius, w_radius)

            mxface = 300
            miface = 25
            mxline = max(obj.width, obj.height)
            gamma = (mxline - miface) / (mxface - miface) * 10
            gamma = min(max(0, gamma), 10) + 1
            preprocess.draw_gaussian(heatmap_posweight[classes, :, :], (cx, cy), posweight_radius, k=gamma)

            range_expand_x = math.ceil(w_radius)
            range_expand_y = math.ceil(h_radius)

            min_expand_size = 3
            range_expand_x = max(min_expand_size, range_expand_x)
            range_expand_y = max(min_expand_size, range_expand_y)

            icx, icy = cx, cy
            reg_landmark = None
            fill_threshold = 0.3
			
            if obj.haslandmark:
                reg_landmark = np.array(obj.x5y5_cat_landmark) / stride
                x5y5 = [cx]*5 + [cy]*5
                rvalue = (reg_landmark - x5y5)
                landmark_gt[0:10, cy, cx] = np.array(preprocess.log(rvalue)) / 4
                landmark_mask[0, cy, cx] = 1

            if not obj.rotate:
                for cx in range(icx - range_expand_x, icx + range_expand_x + 1):
                    for cy in range(icy - range_expand_y, icy + range_expand_y + 1):
                        if cx < fm_width and cy < fm_height and cx >= 0 and cy >= 0:
                            
                            my_gaussian_value = 0.9
                            gy, gx = cy - icy + range_expand_y, cx - icx + range_expand_x
                            if gy >= 0 and gy < gaussian_map.shape[0] and gx >= 0 and gx < gaussian_map.shape[1]:
                                my_gaussian_value = gaussian_map[gy, gx]
                                
                            distance = math.sqrt((cx - icx)**2 + (cy - icy)**2)
                            if my_gaussian_value > fill_threshold or distance <= min_expand_size:
                                already_distance = distance_map[classes, cy, cx]
                                my_mix_distance = (1 - my_gaussian_value) * distance

                                if my_mix_distance > already_distance:
                                    continue

                                distance_map[classes, cy, cx] = my_mix_distance
                                reg_tlrb[classes*4:(classes+1)*4, cy, cx] = reg_box
                                reg_mask[classes, cy, cx] = 1


        return T.to_tensor(image), heatmap_gt, heatmap_posweight, reg_tlrb, reg_mask, landmark_gt, landmark_mask, len(objs), keep_mask



class App(object):
    def __init__(self, labelfile, imagesdir):

        self.width, self.height = 800, 800
        self.mean = [0.408, 0.447, 0.47]
        self.std = [0.289, 0.274, 0.278]
        self.batch_size = 18
        self.lr = 1e-4
        self.gpus = [7] #[0, 1, 2, 3]
        self.gpu_master = self.gpus[0]
        self.model = DBFace()
        self.model.load('/home/data/DBFace/model/dbface.pth')
        #self.model = nn.DataParallel(self.model, device_ids=self.gpus)
        self.model.cuda(device=self.gpu_master)
        self.model.train()

        self.focal_loss = losses.FocalLoss()
        self.giou_loss = losses.GIoULoss()
        self.landmark_loss = losses.WingLoss(w=2)
        self.train_dataset = LDataset(labelfile, imagesdir, mean=self.mean, std=self.std, width=self.width, height=self.height)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.per_epoch_batchs = len(self.train_loader)
        self.iter = 0
        self.epochs = 150

        label_file = '/home/data/Datasets/SD/self_test/label.txt'
        images_dir = '/home/data/Datasets/SD/self_test/images/'
        self.mean = [0.408, 0.447, 0.47]
        self.std = [0.289, 0.274, 0.278]
        self.val_gt = preprocess.load_webface(label_file, images_dir)
        self.reg_tlrb, self.reg_mask = [], []

        for i in range(len(self.val_gt)):
            box, mask = self.eval_reg(self.val_gt[i])
            self.reg_tlrb.append(box)
            self.reg_mask.append(mask)

        self.writer = SummaryWriter(log_dir=f"./output/train_result/{trial_name}/logs/tb_file", comment='dbface')

    def set_lr(self, lr):

        self.lr = lr
        log.info(f"setting learning rate to: {lr}")
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


    def train_epoch(self, epoch):
        
        for indbatch, (images, heatmap_gt, heatmap_posweight, reg_tlrb, reg_mask, landmark_gt, landmark_mask, num_objs, keep_mask) in enumerate(self.train_loader):

            self.iter += 1

            batch_objs = sum(num_objs)
            batch_size = self.batch_size

            if batch_objs == 0:
                batch_objs = 1

            heatmap_gt          = heatmap_gt.to(self.gpu_master)
            heatmap_posweight   = heatmap_posweight.to(self.gpu_master)
            keep_mask           = keep_mask.to(self.gpu_master)
            reg_tlrb            = reg_tlrb.to(self.gpu_master)
            reg_mask            = reg_mask.to(self.gpu_master)
            landmark_gt         = landmark_gt.to(self.gpu_master)
            landmark_mask       = landmark_mask.to(self.gpu_master)
            images              = images.to(self.gpu_master)

            hm, tlrb, landmark  = self.model(images)
            # hm = hm.sigmoid()
            hm = torch.clamp(hm, min=1e-4, max=1-1e-4)
            # tlrb = torch.exp(tlrb)

            hm_loss = self.focal_loss(hm, heatmap_gt, heatmap_posweight, keep_mask=keep_mask) / batch_objs
            reg_loss = self.giou_loss(tlrb, reg_tlrb, reg_mask)*5
            landmark_loss = self.landmark_loss(landmark, landmark_gt, landmark_mask)*0.1
            loss = hm_loss + reg_loss + landmark_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_flt = epoch + indbatch / self.per_epoch_batchs

            if indbatch % 10 == 0:
                self.writer.add_scalar('lr', self.lr, self.iter)
                self.writer.add_scalar('hm_loss', hm_loss, self.iter)
                self.writer.add_scalar('reg_loss', reg_loss, self.iter)
                self.writer.add_scalar('landmark_loss', landmark_loss, self.iter)
                self.writer.add_scalar('loss', loss, epoch)
                log.info(
                    f"iter: {self.iter}, lr: {self.lr:g}, epoch: {epoch_flt:.2f}, loss: {loss.item():.2f}, hm_loss: {hm_loss.item():.2f}, "
                    f"box_loss: {reg_loss.item():.2f}, ldmk_loss: {landmark_loss.item():.5f}"
                )

            if indbatch % 1000 == 0:
                log.info("save hm")
                # hm_image = hm[0, 0].cpu().data.numpy()
                # preprocess.imwrite(f"{jobdir}/imgs/hm_image.jpg", hm_image * 255)
                # preprocess.imwrite(f"{jobdir}/imgs/hm_image_gt.jpg", heatmap_gt[0, 0].cpu().data.numpy() * 255)

                # image = np.clip((images[0].permute(1, 2, 0).cpu().data.numpy() * self.std + self.mean) * 255, 0, 255).astype(np.uint8)
                # outobjs = eval.detect_images_giou_with_netout(hm, tlrb, landmark, threshold=0.1, ibatch=0)
                #
                # im1 = image.copy()
                # for obj in outobjs:
                #     preprocess.drawbbox(im1, obj)
                # preprocess.imwrite(f"{jobdir}/imgs/train_result.jpg", im1)
                with torch.no_grad():
                    self.model.eval()
                    reg_losses = []
                    for i in tqdm(range(len(self.val_gt))):
                        imgfile, _ = self.val_gt[i]
                        img = cv2.imread(imgfile)
                        H, W, _ = img.shape
                        mhw = max(H, W)
                        new_img = np.zeros((mhw, mhw, 3), np.float32)
                        new_img[:H, :W, :] = img
                        img = cv2.resize(new_img, (self.width, self.height))
                        img = ((img / 255. - self.mean) / self.std).astype(np.float32)
                        img = img.transpose(2, 0, 1)
                        img = torch.from_numpy(img)[None].cuda()
                        _, pred_tlrb, _ = self.model(img)
                        reg_loss = self.giou_loss(pred_tlrb, self.reg_tlrb[i], self.reg_mask[i])
                        reg_losses.append(reg_loss)
                    box_val = 1 - torch.Tensor(reg_losses).mean()
                    log.info(f"box_val: {box_val}")
                self.model.train()


    def train(self):

        lr_scheduer = {
            1: 1e-3,
            2: 2e-3,
            3: 1e-3,
            60: 1e-4,
            120: 1e-5
        }

        # train
        self.model.train()
        for epoch in range(self.epochs):

            if epoch in lr_scheduer:
                self.set_lr(lr_scheduer[epoch])

            self.train_epoch(epoch)
            file = f"{jobdir}/models/{epoch + 1}.pth"
            preprocess.mkdirs_from_file_path(file)
            torch.save(self.model.state_dict(), file)
        self.writer.close()

    def eval_reg(self, gt):
        imgfile, objs = gt
        img = cv2.imread(imgfile)
        H, W, _ = img.shape
        mhw = max(H, W)
        scale = mhw / 256
        reg_tlrb = np.zeros((1 * 4, 64, 64), np.float32)
        reg_mask = np.zeros((1, 64, 64), np.float32)

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

        return reg_tlrb, reg_mask

if __name__ == '__main__':

    trial_name = "try1"
    jobdir = f"./output/train_result/{trial_name}"

    log = logger.create(trial_name, f"{jobdir}/logs/{trial_name}.log")
    app = App("/home/data/Datasets/WIDERFace/retinaface_labels/train/label.txt",
              "/home/data/Datasets/WIDERFace/WIDER_train/images")
    app.train()