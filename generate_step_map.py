import time
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import os
from torch.nn.modules.loss import CrossEntropyLoss
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt

from DataSet.OCT_GT_DATA import OCTGTDATA
from tools import generate_map_transform, image_transform, DiceLoss, draw_loss_figure, make_dir, roi_detection_process

def generate_map_gt_process(model, train_scan_path_list, batch_size, epoch, lr, num_classes, gt_type, device, intermittent_list, using_max_map, gt_train_gt, save_path, using_roi):
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_train_decoder_list = []

    for train_decoder_epoch_num in range(epoch):
        print("Training decoder No.%d epoch..." % (train_decoder_epoch_num + 1))
        time.sleep(0.5)
        DATA = OCTGTDATA(train_scan_path_list, gt_type, intermittent_list[train_decoder_epoch_num], image_transform, generate_map_transform)
        data_loader = DataLoader(dataset=DATA, batch_size=batch_size, shuffle=False)
        for input_batch, gt_batch, name_batch, roi_batch in tqdm.tqdm(data_loader):
            # plt.subplot(4, 2, 1)
            # plt.imshow(roi_batch[0][0], "gray")
            # plt.subplot(4, 2, 2)
            # plt.imshow(roi_batch[0][1], "gray")
            # plt.subplot(4, 2, 3)
            # plt.imshow(roi_batch[1][0], "gray")
            # plt.subplot(4, 2, 4)
            # plt.imshow(roi_batch[1][1], "gray")
            # plt.subplot(4, 2, 5)
            # plt.imshow(roi_batch[2][0], "gray")
            # plt.subplot(4, 2, 6)
            # plt.imshow(input_batch[2][1], "gray")
            # plt.subplot(4, 2, 7)
            # plt.imshow(input_batch[3][0], "gray")
            # plt.subplot(4, 2, 8)
            # plt.imshow(input_batch[3][1], "gray")
            # plt.show()
            input_batch, gt_batch, roi_batch = input_batch.to(device), gt_batch.to(device), roi_batch.to(device)
            if gt_train_gt:
                decoder1, decoder2, decoder3, decoder4, outputs = model(gt_batch)
            else:
                if using_roi:
                    decoder1, decoder2, decoder3, decoder4, outputs = model(roi_batch)
                else:
                    decoder1, decoder2, decoder3, decoder4, outputs = model(input_batch)
            loss_train_decoder_dice = dice_loss(outputs, gt_batch.squeeze(1), softmax=True)
            loss_train_decoder_ce = ce_loss(outputs, gt_batch[:].long().squeeze(1))
            loss_train_decoder = 0.4 * loss_train_decoder_ce + 0.6 * loss_train_decoder_dice

            optimizer.zero_grad()
            loss_train_decoder.backward()
            optimizer.step()

            loss_train_decoder_list.append(loss_train_decoder.item())

    save_model_path = os.path.join(save_path, 'train_decoder_parameter.pth')
    torch.save(model.state_dict(), save_model_path)
    save_train_decoder_loss_path = os.path.join(save_path, 'train_decoder_loss.png')
    draw_loss_figure(save_train_decoder_loss_path, loss_train_decoder_list)

    DATA_generate = OCTGTDATA(train_scan_path_list, gt_type, transform_output=image_transform, transform_input=generate_map_transform, generating=True)
    test_loader = DataLoader(dataset=DATA_generate, batch_size=1, shuffle=False)
    model.eval()
    save_map_path = os.path.join(save_path, 'map')
    make_dir(save_map_path)
    print("Generating the map")
    time.sleep(0.5)
    with torch.no_grad():
        for input_tensor, gt, name, roi_tensor in tqdm.tqdm(test_loader):
            now_name = name[0].split('.')[0]
            now_save_map_path = os.path.join(save_map_path, now_name)
            input_tensor, gt, roi_tensor = input_tensor.to(device), gt.to(device), roi_tensor.to(device)
            if gt_train_gt:
                decoder1, decoder2, decoder3, decoder4, output = model(gt)
            else:
                if using_roi:
                    decoder1, decoder2, decoder3, decoder4, outputs = model(roi_tensor)
                else:
                    decoder1, decoder2, decoder3, decoder4, output = model(input_tensor)

            now_save_map_decoder1_path = now_save_map_path + '_decoder1'
            decoder1_array = decoder1.squeeze().detach().cpu().numpy()
            if using_max_map:
                decoder1_array = np.max(decoder1_array, 0)
            np.save(now_save_map_decoder1_path, decoder1_array)

            now_save_map_decoder2_path = now_save_map_path + '_decoder2'
            decoder2_array = decoder2.squeeze().detach().cpu().numpy()
            if using_max_map:
                decoder2_array = np.max(decoder2_array, 0)
            np.save(now_save_map_decoder2_path, decoder2_array)

            now_save_map_decoder3_path = now_save_map_path + '_decoder3'
            decoder3_array = decoder3.squeeze().detach().cpu().numpy()
            if using_max_map:
                decoder3_array = np.max(decoder3_array, 0)
            np.save(now_save_map_decoder3_path, decoder3_array)

            now_save_map_decoder4_path = now_save_map_path + '_decoder4'
            decoder4_array = decoder4.squeeze().detach().cpu().numpy()
            if using_max_map:
                decoder4_array = np.max(decoder4_array, 0)
            np.save(now_save_map_decoder4_path, decoder4_array)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return save_map_path