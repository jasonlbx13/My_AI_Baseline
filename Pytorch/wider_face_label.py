 # -*- coding: UTF-8 -*-

# @Time         : 2020/2/6
# @Author       : ZP
# @File         : wider_face_label.py
# @Description  :

import cv2
import os
import argparse
from collections import OrderedDict
import sys


class Getoutofloop(Exception):
    pass


# def load_file(img_set="train"):
# def load_file(label_path):
#     img_boxes_dict = OrderedDict()
#     # gtfilepath = os.path.join(retinaface_gt_file_dir, img_set, "label.txt")
#     gtfilepath = label_path
#     with open(gtfilepath, 'r') as gtfile:
#         index = 0
#         while True:  # and len(faces)<10
#             line = gtfile.readline().strip()
#             if line == "":
#                 if len(bboxes) != 0:
#                     imgfilepath = filename  # [:-4]
#                     img_boxes_dict[imgfilepath] = bboxes
#                     print("end!")
#                 break
#             if line.startswith("#"):
#                 if index != 0:
#                     if len(bboxes) != 0:
#                         imgfilepath = filename  # [:-4]
#                         img_boxes_dict[imgfilepath] = bboxes
#                     else:
#                         print("{} no face".format(filename))
#
#                 filename = line[1:].strip()
#                 # print(("\r" + str(index) + ": " + filename + "\t"))
#                 index = index + 1
#                 bboxes = []
#                 continue
#             else:
#                 line = [float(x) for x in line.strip().split()]
#                 if len(line) == 0:
#                     break
#                 if int(line[3]) <= 0 or int(line[2]) <= 0:
#                     print("{} width or height is 0".format(filename))
#                     continue
#                 bboxes.append(line)
#     return img_boxes_dict


def load_file(label_file):
    with open(label_file, 'r') as f1:
        data = f1.readlines()

    name_set = set()
    res_dict = OrderedDict()
    name = ''
    for line in data:
        line = line.strip('\n')
        if line.startswith('#'):
            name = line[2:]
            if name not in name_set:
                name_set.add(name)
                res_dict[name] = []
            else:
                pass
        else:
            if name=='':
                print ('请检查标注文件格式是否正确')
                break
            res_dict[name].append(list(map(float, line.split(' '))))

    return res_dict


def save_refine_file(img_lines_dict, label_file_path, img_set="train", filename="label_refine.txt"):
    # save_file_path = os.path.join(retinaface_gt_file_dir, img_set, filename)
    label_file_dir = os.path.dirname(label_file_path)
    save_file_path = os.path.join(label_file_dir, filename)
    save_file = open(save_file_path, 'w')
    for img_name,  lines in img_lines_dict.items():
        tmp_str = "# " + img_name + "\n"
        for i in range(len(lines)):
            tmp_str += " ".join(str(x) for x in lines[i])
            tmp_str += "\n"
        # print(tmp_str)
        save_file.write(tmp_str)
        save_file.flush()
    save_file.close()


# def show_images_with_key_actions(image_root, img_set):
def show_images_with_key_actions(args):
    res = load_file(args.label_file_path)
    total_num = len(res.keys())
    print("total images ", total_num)
    keys = [key for key in res.keys()]
    res_new = OrderedDict()
    face_name_set = set()

    i = 0
    while i < total_num:
        image_name = keys[i]
        i += 1
        print("current index: {}/{}, {}".format(i, total_num, image_name))
        bboxes = res[image_name]
        bboxes_new = []
        stack = []
        # img = cv2.imread(os.path.join(image_root, image_name + ".jpg"))
        if image_name.endswith(".jpg"):
            img = cv2.imread(os.path.join(args.img_dir, image_name))
        else:
            img = cv2.imread(os.path.join(args.img_dir, image_name + ".jpg"))
        pre_flag = -1
        cur_flag = -1
        cv2.namedWindow("test", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        try:
            j = 0
            while j < len(bboxes):
                face_name = image_name+str(j)
                bbox = bboxes[j]
                tmp = bboxes[j].copy()
                print(f'face_num: {j}, labeled_face: {len(face_name_set)}')
                while True:
                    img_c = img.copy()
                    h, w, _ = img.shape
                    xb, yb, wb, hb = bbox[0:4]
                    # landmark = bbox[4:]
                    xb = int(xb)
                    yb = int(yb)
                    wb = int(wb)
                    hb = int(hb)
                    if face_name not in face_name_set:
                        if 20<wb<110 and 20<hb<110:
                            face_name_set.add(face_name)
                        else:
                            break
                    if stack == [] or j != stack[-1]:
                        stack.append(j)

                    x1 = xb - 50
                    y1 = yb - 50
                    x2 = xb + wb + 50
                    y2 = yb + hb + 50
                    if x1 < 0:
                        x1 = 0
                    if y1 < 0:
                        y1 = 0
                    if x2 > w - 1:
                        x2 = w - 1
                    if y2 > h - 1:
                        y2 = h - 1
                    img_re = img_c[y1: y2, x1: x2, :]
                    cv2.rectangle(img_re, (int(xb - x1), int(yb - y1)), (int(xb - x1 + wb), int(yb - y1 + hb)),
                                  (0, 0, 255), thickness=1)
                    cv2.putText(img_re, "{}x{}".format(wb, hb), (int(xb - x1 - 5), int(yb - y1 -5)), cv2.FONT_HERSHEY_PLAIN,
                                1, color=(255, 0, 255))
                    cv2.putText(img_re, image_name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 255))
                    # for idx in range(5):
                    #     cv2.circle(img_re, (int(landmark[3*idx]) - x1, int(landmark[3*idx+1]) - y1), 2, (255, 0, 255), -1)
                    cv2.imshow("test", img_re)
                    flag = cv2.waitKey(0)
                    if flag == 27:
                        sys.exit()
                    if flag == 32:
                        break  # 空格
                    if flag > -1 and flag != pre_flag:  # whether press other key
                        cur_flag = flag
                    pre_flag = flag

                    # response event
                    if cur_flag == ord('a'):
                        # adjust bbox left frame
                        bbox[0] -= 1
                        bbox[2] += 1
                    elif cur_flag == ord('d'):
                        bbox[0] += 1
                        bbox[2] -= 1
                    elif cur_flag == ord('w'):
                        # adjust bbox right frame
                        bbox[1] -= 1
                        bbox[3] += 1
                    elif cur_flag == ord('s'):
                        bbox[1] += 1
                        bbox[3] -= 1
                    elif cur_flag == ord('4'):
                        # adjust bbox up frame
                        bbox[2] -= 1
                    elif cur_flag == ord('6'):
                        # adjust bbox bottom frame
                        bbox[2] += 1
                    elif cur_flag == ord('8'):
                        bbox[3] -= 1
                    elif cur_flag == ord("5"):
                        bbox[3] += 1
                    elif cur_flag == ord("q"):
                        bbox[0] = max(0, bbox[0]-5)
                        bbox[1] = max(0, bbox[1]-5)
                        bbox[2] += 10
                        bbox[3] += 10
                    elif cur_flag == ord("e"):
                        bbox[0] += 5
                        bbox[1] += 5
                        bbox[2] = max(0, bbox[2]-10)
                        bbox[3] = max(0, bbox[3]-10)
                    elif cur_flag == ord("t"):
                        bbox[0] += int(0.5*bbox[2])-1
                        bbox[1] += int(0.5*bbox[3])-1
                        bbox[2] = 3
                        bbox[3] = 3
                    elif cur_flag == ord("r"):
                        bbox[1] = max(0, bbox[1]-1)
                    elif cur_flag == ord("f"):
                        bbox[1] += 1
                    elif cur_flag == ord("c"):
                        bbox[0] = max(0, bbox[0]-1)
                    elif cur_flag == ord("v"):
                        bbox[0] += 1
                    elif cur_flag == ord("z"):
                        bbox = tmp.copy()
                    elif cur_flag == ord('n'):
                        save_refine_file(res, args.label_file_path)
                        save_refine_file(res_new, args.label_file_path, filename="label_refine_new.txt")
                        print("save finish")
                    elif cur_flag == ord('j'):
                        print("input a number: ")
                        line = input()
                        try:
                            num = int(line)
                        except Exception:
                            print("Input is not a number")
                            continue
                        if num > total_num - 1:
                            print("The number is bigger than the annos length")
                            num = i
                        i = num - 1
                        # save_refine_file(res, img_set)
                        # save_refine_file(res_new, img_set, "label_refine_new.txt")
                        save_refine_file(res, args.label_file_path)
                        save_refine_file(res_new, args.label_file_path, filename="label_refine_new.txt")
                        raise Getoutofloop
                    elif cur_flag == ord("k"):
                        print("input a image name: ")
                        line = input()
                        try:
                            line = str(line).strip().strip("\n")
                        except Exception:
                            print('Not valid image name')
                            continue
                        save_refine_file(res, args.label_file_path)
                        save_refine_file(res_new, args.label_file_path, filename="label_refine_new.txt")
                        try:
                            line_idx = list(res.keys()).index(line)
                        except ValueError:
                            print("{} is not in list".format(line))
                            continue
                        print("input image name index: ", line_idx)
                        i = line_idx
                        raise Getoutofloop
                    elif cur_flag == ord("b"):
                        bboxes[j] = bbox
                        # j = j - 2
                        stack.pop()
                        if stack != []:
                            j = stack.pop()-1
                        else:
                            j = 0
                        break
                    else:
                        print('...')
                if bbox[0] < 0:
                    bbox[0] = 0
                if bbox[1] < 0:
                    bbox[1] = 0
                if bbox[2] > w - 1:
                    bbox[2] = w - 1
                if bbox[3] > h - 1:
                    bbox[3] = h - 1
                if cur_flag != ord('b'):
                    bboxes[j] = bbox
                    if face_name in face_name_set:
                        if j < len(bboxes_new):
                            bboxes_new[j] = bbox
                        else:
                            bboxes_new.append(bbox)
                res[image_name] = bboxes
                if bboxes_new != []:
                    res_new[image_name] = bboxes_new
                if j % 200 == 0:
                    save_refine_file(res, args.label_file_path)
                    save_refine_file(res_new, args.label_file_path, filename="label_refine_new.txt")
                j = j + 1
                if j < 0:
                    j = 0
        except Getoutofloop:
            print("current jump index: ", i)
            pass


def get_args():
    parser = argparse.ArgumentParser(description="detect single image")
    parser.add_argument("--img_dir", type=str, help="image dir")
    parser.add_argument("--label_file_path", type=str, help="label file path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    show_images_with_key_actions(args)

