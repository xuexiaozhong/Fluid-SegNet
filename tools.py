import gc
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from settings import IMAGE_SIZE
from Networks.UNet import UNet

def make_dir(dir):
    if os.path.exists(dir) is False:
        os.makedirs(dir)

def image_padding(image):
    h, w = image.shape
    assert IMAGE_SIZE[1] >= h or IMAGE_SIZE[0] >= w, \
        f"The IMAGE_SIZE ({IMAGE_SIZE[1]}*{IMAGE_SIZE[0]}) is smaller than input image size ({h}*{w})."
    top = (IMAGE_SIZE[1] - h) // 2
    bottom = (IMAGE_SIZE[1] - h) - top
    left = (IMAGE_SIZE[0] - w) // 2
    right = (IMAGE_SIZE[0] - w) - left
    padded_image = cv2.copyMakeBorder(image, top=top, bottom=bottom, left=left, right=right,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])
    image_array = np.array(padded_image, dtype=np.float32)
    image_array = image_array / 255

    return image_array

def image_transform(image):
    image_array = image_padding(image)
    image_tensor = torch.tensor(image_array)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def generate_map_transform(scan, gt, intermittent):
    padded_scan_array = image_padding(scan)
    padded_gt_array = image_padding(gt)
    blank_gt_array = image_padding(np.zeros_like(gt))
    tensor_array = np.zeros((2, IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.float32)
    tensor_array[0, :, :] = padded_scan_array
    if intermittent == 1:
        tensor_array[1,:,:] = padded_gt_array
    elif intermittent == 0:
        tensor_array[1,:,:] = blank_gt_array
    else:
        raise ValueError('Illegal intermittent data type (%s)!' % intermittent)

    tensor = torch.tensor(tensor_array)
    return tensor

def image_transform_3d(scan, before_scan, after_scan):
    padded_scan_array = image_padding(scan)
    padded_before_scan_array = image_padding(before_scan)
    padded_after_scan_array = image_padding(after_scan)

    tensor_array = np.zeros((3, IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.float32)
    tensor_array[0,:,:] = padded_before_scan_array
    tensor_array[1,:,:] = padded_scan_array
    tensor_array[2,:,:] = padded_after_scan_array

    tensor = torch.tensor(tensor_array)
    return tensor

def Intermittent_list_generate(train_scan_path_list, train_decoder_epoch, with_gt_group, using_intermittent_map):
    length = len(train_scan_path_list)
    random_index_list = range(0, length)
    random_index_list = random.sample(random_index_list, length)
    every_epoch_true_length_list = []
    for i in range(train_decoder_epoch):
        if i < with_gt_group - 1:
            every_epoch_true_length_list.append((length // with_gt_group) * (i+1))
        else:
            every_epoch_true_length_list.append(length)

    intermittent_list = []
    for i in range(train_decoder_epoch):
        j = i % with_gt_group
        if using_intermittent_map:
            sub_intermittent_array = np.zeros(length)
            if j == 0:
                for k in random_index_list[0:every_epoch_true_length_list[j]]:
                    sub_intermittent_array[k] = 1
            else:
                for k in random_index_list[every_epoch_true_length_list[j-1]:every_epoch_true_length_list[j]]:
                    sub_intermittent_array[k] = 1

            intermittent_list.append(sub_intermittent_array)
        else:
            sub_intermittent_array = np.ones(length)
            intermittent_list.append(sub_intermittent_array)
    return intermittent_list

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor): # if input [4,1,496,1024]
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()  # then output [4,2,1,496,1024]

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class RoiDetectData(Dataset):
    def __init__(self, image_path_list, transform=None):
        self.image_path_list = image_path_list
        self.transform = transform

    def __len__(self):
        self.length = len(self.image_path_list)

        return self.length

    def __getitem__(self, item):
        image_path = self.image_path_list[item]
        image = cv2.imread(image_path, 0)
        image_tensor = self.transform(image)

        return image_tensor

def roi_detection_process(image_path):
    image_original = cv2.imread(image_path, 0)
    non_zero_num = len(np.flatnonzero(image_original))
    if non_zero_num == 0:
        return image_original
    else:
        pth_path = './Networks/parameter.pth'
        model = UNet(in_channels=1, num_classes=2, bilinear=True, base_c=32)
        model.load_state_dict(torch.load(pth_path))
        model.eval()

        image_path_list = []
        image_path_list.append(image_path)
        DATA = RoiDetectData(image_path_list, image_transform)
        loader = DataLoader(dataset=DATA, batch_size=1, shuffle=False)

        out_list = []
        for image in loader:
            output = model(image)
            out = torch.argmax(torch.softmax(output[-1], dim=1), dim=1).squeeze(0)
            out = out.detach().numpy()
            out_binary_uint8 = np.array(out*255, dtype=np.uint8)
            out_max = np.zeros_like(out_binary_uint8)

            contours, _ = cv2.findContours(out_binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            area_list = []
            for i in range(len(contours)):
                area_list.append(cv2.contourArea(contours[i]))
            max_idx = np.argmax(np.array(area_list))
            out_max = cv2.drawContours(out_max, contours, max_idx, 255, -1)
            out_list.append(out_max)
        if len(out_list) != 1:
            raise ValueError('The error when doing roi detection!')
        out_result = out_list[0]
        image_original[out_result == 0] = 0

        del model
        gc.collect()
        torch.cuda.empty_cache()
        return image_original

def draw_loss_figure(path, loss_list):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    iter_times = len(loss_list)

    ax.plot(range(0, iter_times), loss_list)
    ax.set_xlabel('Iter_times')
    ax.set_ylabel('Loss')

    ax.set_title("Iteration_loss")

    plt.xlim(0, iter_times)
    plt.ylim(0.0, 1.0)

    plt.savefig(path)
    plt.close()

def update(model, index):
    parameters = []
    for name, p in model.named_parameters():
        if 'in_conv' in name:
            parameters.append(p)
        elif 'down' in name:
            parameters.append(p)
        elif 'fusion' in name:
            parameters.append(p)
        else:
            if 'up' in name:
                if int(name.split('.')[0].split('p')[-1]) < index + 1:
                    parameters.append(p)
            if index == 4 and 'out_conv' in name:
                parameters.append(p)
    return parameters

def cal_metrics(out, gt):
    over = out + gt
    intersection = len(np.flatnonzero(over == 2))
    un_intersection = len(np.flatnonzero(over == 1))
    out_num = len(np.flatnonzero(out == 1))
    gt_num = len(np.flatnonzero(gt == 1))
    IoU = intersection / (intersection + un_intersection + 1e-9)
    DICE = (2 * intersection) / (out_num + gt_num + 1e-9)
    return IoU, DICE

def save_image(path, image, out, gt, IoU, name):
    case_name = name[0].split('.')[0]
    image_uint8 = np.array(image * 255, dtype=np.uint8)
    image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    out_binary_uint8 = np.array(out * 255, dtype=np.uint8)
    gt_binary_uint8 = np.array(gt * 255, dtype=np.uint8)

    contours_gt, _ = cv2.findContours(gt_binary_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_out, _ = cv2.findContours(out_binary_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours_gt)):
        image_uint8 = cv2.drawContours(image_uint8, contours_gt, i, (0, 0, 255), 1)
    for j in range(len(contours_out)):
        image_uint8 = cv2.drawContours(image_uint8, contours_out, j, (0, 255, 0), 1)

    image_name = case_name + '_' + str(IoU) + '.png'
    out_name = case_name + '_' + 'OUT' + '.png'
    image_path = os.path.join(path, image_name)
    out_path = os.path.join(path, out_name)

    cv2.imwrite(image_path, image_uint8)
    cv2.imwrite(out_path, out_binary_uint8)

def save_txt(path, IoU_list, DICE_list, batch_size, learning_rate, IMG_SIZE,
             train_decoder_epoch, train_each_map_epoch, train_final_epoch,
             gt_type):
    txt_path = os.path.join(path, 'result.txt')
    file = open(txt_path, 'w')
    file.writelines(["GT type: ", str(gt_type), "\t",
                     "Batch num: ", str(batch_size), "\t",
                     "LR: ", str(learning_rate), "\t",
                     "Train decoder epoch: ", str(train_decoder_epoch), "\t",
                     "Train each map epoch: ", str(train_each_map_epoch), "\n",
                     "Train final epoch: ", str(train_final_epoch), "\t",
                     "Image size: ", str(IMG_SIZE), "\n"])

    file.writelines(["The average IoU is: ", str(np.mean(np.array(IoU_list))), "\n"])
    file.writelines(["The average DICE is: ", str(np.mean(np.array(DICE_list))), "\n"])
    file.write("////////////////////////////////////////////////\n")

    for i in range(len(IoU_list)):
        file.writelines(["The average IoU of subset", str(i), "is: ", str(IoU_list[i]), "\n"])
        file.writelines(["The average DICE of subset", str(i), "is: ", str(DICE_list[i]), "\n"])
        file.write("-------------------------------------------------\n")
    file.close()