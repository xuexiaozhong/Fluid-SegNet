import time
import os
import numpy as np
import torch

from settings import *
from tools import make_dir, save_txt, Intermittent_list_generate
from DataSet.generator import generate_dataset
from DataSet.scan_generator import generate_scan_dataset
from DataSet.generator_subset import generate_subset_dataset
from Networks.UNet import UNet
from Networks.YNet import YNet
from Networks.WNet import DS_DC_YNet
from Networks.SE_Unet import SE_UNet
from Networks.Dilated_YNet import Dilated_Y_Net
from Networks.Unetpp import NestedUNet
from Networks.CBAM_Dilated_Ynet import CBAM_Dilated_Y_Net
from generate_step_map import generate_map_gt_process
from train_each_map import train_each_map_process, train_each_map_process_without_multiloss
from test import test_process

date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
print("Date is: ", date)
print("Train decoder epochs: %d, Train each map epoch: %d, Train final epoch: %d"
      "Learning rate: %s, Image size: %s, \nBatch size: %d, Basic channel of U-Net: %d, "
      "Num of classes: %d, GT type: %s, Fold num: %d, \n"
      "Using multi loss function: %s, Using intermittent generating map: %s"
      % (TRAIN_DECODER_EPOCH, TRAIN_EACH_MAP_EPOCH, TRAIN_FINAL_EPOCH, LR, IMAGE_SIZE,
         BATCH_SIZE, BASIC_CHANNEL, NUM_CLASSES, GT_TYPE, FOLD_NUM, USING_MULTI_LOSS, USING_INTERMITTENT_MAP))

now_save_path = os.path.join(SAVE_PATH, date)
make_dir(now_save_path)

shuffled_AMD, shuffled_DME, AMD_project_list, DME_project_list = generate_dataset(DATA_PATH)

mean_IoU_list = []
mean_Dice_list = []
if USING_MULTI_LOSS:
    for i in range(FOLD_NUM):
        print("The No.%d fold experiments" % (i))
        time.sleep(0.5)
        save_subset_path = os.path.join(now_save_path, str(i))
        make_dir(save_subset_path)
        save_train = os.path.join(save_subset_path, 'Train')
        make_dir(save_train)
        save_test = os.path.join(save_subset_path, 'Test')
        make_dir(save_test)

        train_project_list, test_project_list = generate_subset_dataset(AMD_project_list, DME_project_list, AMD_SUBSET_LEN, DME_SUBSET_LEN, i)
        train_scan_path_list, test_scan_path_list = generate_scan_dataset(train_project_list, test_project_list)
        intermittent_list = Intermittent_list_generate(train_scan_path_list, TRAIN_DECODER_EPOCH, WITH_GT_GROUP, USING_INTERMITTENT_MAP)
        # print(len(intermittent_list))
        # for m in range(len(intermittent_list)):
        #     print("the No.%d intermittent_list" % m)
        #     num_1 = len(np.flatnonzero(np.array(intermittent_list[m])))
        #     print("the 1 numbers is: %d" % num_1)
        #     print(len(intermittent_list[m]))
        if GT_TRAIN_GT:
            model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=True, base_c=BASIC_CHANNEL).to(DEVICE)
        else:
            model = UNet(in_channels=2, num_classes=NUM_CLASSES, bilinear=True, base_c=BASIC_CHANNEL).to(DEVICE)

        save_map_path = generate_map_gt_process(model, train_scan_path_list, BATCH_SIZE, TRAIN_DECODER_EPOCH, LR, NUM_CLASSES, GT_TYPE, DEVICE, intermittent_list, USING_MAX_MAP, GT_TRAIN_GT, save_train, USING_ROI_DETECTION)

        if MODEL_TYPE == 'W-Net':
            model_Ynet = DS_DC_YNet(in_channels_2d=1, in_channels_3d=3, num_classes=NUM_CLASSES, base_c=BASIC_CHANNEL, using_max_map=USING_MAX_MAP).to(DEVICE)
        elif MODEL_TYPE == 'Y-Net':
            model_Ynet = YNet(in_channels_2d=1, in_channels_3d=3, num_classes=NUM_CLASSES, base_c=BASIC_CHANNEL, using_max_map=USING_MAX_MAP).to(DEVICE)
        elif MODEL_TYPE == 'Dilated_YNet':
            model_Ynet = Dilated_Y_Net(in_channels_2d=1, in_channels_3d=3, num_classes=NUM_CLASSES, base_c=BASIC_CHANNEL, using_max_map=USING_MAX_MAP).to(DEVICE)
        elif MODEL_TYPE == 'CBAM_Dilated_YNet':
            model_Ynet = CBAM_Dilated_Y_Net(in_channels_2d=1, in_channels_3d=3, num_classes=NUM_CLASSES, base_c=BASIC_CHANNEL, using_max_map=USING_MAX_MAP).to(DEVICE)
        else:
            raise ValueError('No this model!')

        trained_map_parameter_path = None

        for i in range(5):
            trained_map_parameter_path = train_each_map_process(model_Ynet, save_map_path, train_scan_path_list, BATCH_SIZE, TRAIN_EACH_MAP_EPOCH, LR, NUM_CLASSES, GT_TYPE, DEVICE, save_train, i, trained_map_parameter_path, GT_MAP_NORMALIZATION, USING_ROI_DETECTION)

        mean_IoU, mean_Dice = test_process(model_Ynet, trained_map_parameter_path, test_scan_path_list, GT_TYPE, DEVICE, save_test, USING_ROI_DETECTION)

        mean_IoU_list.append(mean_IoU)
        mean_Dice_list.append(mean_Dice)

else:
    for i in range(FOLD_NUM):
        print("The No.%d fold experiments" % (i))
        time.sleep(0.5)
        save_subset_path = os.path.join(now_save_path, str(i))
        make_dir(save_subset_path)
        save_train = os.path.join(save_subset_path, 'Train')
        make_dir(save_train)
        save_test = os.path.join(save_subset_path, 'Test')
        make_dir(save_test)

        train_project_list, test_project_list = generate_subset_dataset(AMD_project_list, DME_project_list,
                                                                        AMD_SUBSET_LEN, DME_SUBSET_LEN, i)
        train_scan_path_list, test_scan_path_list = generate_scan_dataset(train_project_list, test_project_list)

        if MODEL_TYPE == 'W-Net':
            model_Ynet = DS_DC_YNet(in_channels_2d=1, in_channels_3d=3, num_classes=NUM_CLASSES, base_c=BASIC_CHANNEL).to(DEVICE)
        elif MODEL_TYPE == 'Y-Net':
            model_Ynet = YNet(in_channels_2d=1, in_channels_3d=3, num_classes=NUM_CLASSES, base_c=BASIC_CHANNEL).to(DEVICE)
        elif MODEL_TYPE == 'SE-Unet':
            model_Ynet = SE_UNet(in_channels=1, num_classes=NUM_CLASSES, base_c=BASIC_CHANNEL, ratio=8).to(DEVICE)
        elif MODEL_TYPE == 'Dilated_YNet':
            model_Ynet = Dilated_Y_Net(in_channels_2d=1, in_channels_3d=3, num_classes=NUM_CLASSES, base_c=BASIC_CHANNEL).to(DEVICE)
        elif MODEL_TYPE == 'U_Net':
            model_Ynet = UNet(in_channels=1, num_classes=NUM_CLASSES, base_c=BASIC_CHANNEL).to(DEVICE)
        elif MODEL_TYPE == 'U_Net_pp':
            model_Ynet = NestedUNet(num_classes=NUM_CLASSES, input_channels=1, deep_supervision=True).to(DEVICE)
        elif MODEL_TYPE == 'CBAM_Dilated_Ynet':
            model_Ynet = CBAM_Dilated_Y_Net(in_channels_2d=1, in_channels_3d=3, num_classes=NUM_CLASSES, base_c=BASIC_CHANNEL).to(DEVICE)
        else:
            raise ValueError('No This Model!')

        parameter_path = train_each_map_process_without_multiloss(model_Ynet, train_scan_path_list,
                                                                  BATCH_SIZE, TRAIN_EACH_MAP_EPOCH, LR, NUM_CLASSES,
                                                                  GT_TYPE, DEVICE, save_train, USING_ROI_DETECTION)

        mean_IoU, mean_Dice = test_process(model_Ynet, parameter_path, test_scan_path_list, GT_TYPE, DEVICE,
                                           save_test, USING_ROI_DETECTION)

        mean_IoU_list.append(mean_IoU)
        mean_Dice_list.append(mean_Dice)

save_txt(now_save_path, mean_IoU_list, mean_Dice_list, BATCH_SIZE, LR, IMAGE_SIZE, TRAIN_DECODER_EPOCH, TRAIN_EACH_MAP_EPOCH, TRAIN_FINAL_EPOCH, GT_TYPE)