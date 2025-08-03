import time
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm
import os
from torch.nn.modules.loss import CrossEntropyLoss, SmoothL1Loss
import torch
import gc
import numpy as np
import torch.nn.functional as F

from DataSet.OCT_DATA import OCTDATA, OCTDATA_test
from tools import image_transform, image_transform_3d, DiceLoss, draw_loss_figure, update, roi_detection_process

def train_each_map_process(model, map_path, train_scan_path_list, batch_size, train_each_map_epoch, lr, num_classes, gt_type, device, save_path, index, trained_map_parameter_path, gt_map_normalization, using_roi):
    TRAIN_DATA = OCTDATA(train_scan_path_list, map_path, gt_type, index, image_transform, image_transform_3d, gt_map_normalization)
    train_loader = DataLoader(dataset=TRAIN_DATA, batch_size=batch_size, shuffle=False)

    if trained_map_parameter_path is not None:
        model.load_state_dict(torch.load(trained_map_parameter_path))
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    s1_loss = SmoothL1Loss()

    loss_map_list = []
    parameters = update(model, index)
    optimizer = optim.Adam(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    for train_each_map_epoch_num in range(train_each_map_epoch):
        print("Training map %d in No.%d epoch..." %(index+1, train_each_map_epoch_num+1))
        time.sleep(0.5)
        for scan_batch, scan_3d_batch, label_batch, name_batch, decoder_batch, scan_roi_batch, scan_3d_roi_batch in tqdm.tqdm(train_loader):
            # plt.subplot(4, 2, 1)
            # plt.imshow(scan_batch[0][0], "gray")
            # plt.subplot(4, 2, 2)
            # plt.imshow(scan_3d_batch[0][0], "gray")
            # plt.subplot(4, 2, 3)
            # plt.imshow(scan_3d_batch[0][1], "gray")
            # plt.subplot(4, 2, 4)
            # plt.imshow(scan_3d_batch[0][2], "gray")
            # plt.subplot(4, 2, 5)
            # plt.imshow(scan_roi_batch[0][0], "gray")
            # plt.subplot(4, 2, 6)
            # plt.imshow(scan_3d_roi_batch[0][0], "gray")
            # plt.subplot(4, 2, 7)
            # plt.imshow(scan_3d_roi_batch[0][1], "gray")
            # plt.subplot(4, 2, 8)
            # plt.imshow(scan_3d_roi_batch[0][2], "gray")
            # plt.show()

            scan_batch, scan_3d_batch, label_batch, decoder_batch, scan_roi_batch, scan_3d_roi_batch = scan_batch.to(device), scan_3d_batch.to(device), label_batch.to(device), decoder_batch.to(device), scan_roi_batch.to(device), scan_3d_roi_batch.to(device)

            if using_roi:
                outputs = model(scan_roi_batch, scan_3d_roi_batch)
            else:
                outputs = model(scan_batch, scan_3d_batch)

            if index < 4:
                loss = s1_loss(outputs[index], decoder_batch)
            else:
                loss_ce = ce_loss(F.sigmoid(outputs[index]), label_batch[:].long().squeeze(1))
                loss_dice = dice_loss(outputs[index], label_batch.squeeze(1), softmax=True)
                loss = 0.4*loss_ce + 0.6*loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_map_list.append(loss.item())

        scheduler.step()
        torch.cuda.empty_cache()

    save_model_path = os.path.join(save_path, 'train_map_parameter') + '_' + str(index + 1) + '.pth'
    torch.save(model.state_dict(), save_model_path)
    now_save_figure_path = os.path.join(save_path, 'map_') + str(index + 1) + '_epoch_loss.png'
    draw_loss_figure(now_save_figure_path, loss_map_list)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return save_model_path

def train_each_map_process_without_multiloss(model, train_scan_path_list, batch_size, train_each_map_epoch, lr, num_classes, gt_type, device, save_path, using_roi):
    TRAIN_DATA = OCTDATA_test(train_scan_path_list, gt_type, image_transform, image_transform_3d)
    train_loader = DataLoader(dataset=TRAIN_DATA, batch_size=batch_size, shuffle=False)
    epoch = 5*train_each_map_epoch
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    loss_list = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch_num in range(epoch):
        print("Training without multi loss in No.%d epoch..." %(epoch_num+1))
        time.sleep(0.5)
        for scan_batch, scan_3d_batch, label_batch, name_batch, scan_roi_batch, scan_3d_roi_batch in tqdm.tqdm(train_loader):
            scan_batch, scan_3d_batch, label_batch, scan_roi_batch, scan_3d_roi_batch = scan_batch.to(device), scan_3d_batch.to(device), label_batch.to(device), scan_roi_batch.to(device), scan_3d_roi_batch.to(device)

            if using_roi:
                outputs = model(scan_roi_batch, scan_3d_roi_batch)
            else:
                outputs = model(scan_batch, scan_3d_batch)

            loss_ce = ce_loss(outputs[-1], label_batch[:].long().squeeze(1))
            loss_dice = dice_loss(outputs[-1], label_batch.squeeze(1), softmax=True)
            loss = 0.4*loss_ce + 0.6*loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        scheduler.step()
        torch.cuda.empty_cache()

    save_model_path = os.path.join(save_path, 'parameter.pth')
    torch.save(model.state_dict(), save_model_path)
    now_save_figure_path = os.path.join(save_path, 'epoch_loss.png')
    draw_loss_figure(now_save_figure_path, loss_list)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return save_model_path