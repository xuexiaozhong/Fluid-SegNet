import os
import time

from settings import *
from tools import make_dir, save_txt, save_all_txt

from Network.Unet import UNet
from Network.Swin_Unet import SwinUnet
from Network.NAT_Unet_Mizuno import NAT_UNET as NAT_Unet_Mizuno
from Network.NAT_Unet_Xue import NAT_Unet as NAT_Unet_Xue
from Network.NAT_Unet_Multi_Scale import NAT_Multi_Scale_UNET
from Network.NAT_Skip_Unet import NAT_Skip_Unet
from Network.NAT_Unet_Multi_Scale_Output import MSO_NAT
from Dataset.generator import generate_dataset
from train import train_process
from test import test_process

date = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
print("Date is: ", date)
print("*"*50)
print("Epoch num: %d, Batch num: %d, LR: %s, Image size: %s, Basic channel: %s, "
      "Model type: %s,\nRatio list: %s, Experiment num: %d, GT type: %s, Num classes: %d, "
      "Input channel: %d"
      %(EPOCHS, BATCH_SIZE, LR, IMAGE_SIZE, BASIC_CHANNEL, MODEL_TYPE, RATIO_LIST, EXPERIMENT_NUM, GT_TYPE, NUM_CLASSES, INPUT_CHANNEL))
print("*"*50)

Now_Save_Path = SAVE_PATH + date + '_' + MODEL_TYPE + '_' + GT_TYPE
make_dir(Now_Save_Path)
save_path_train = os.path.join(Now_Save_Path, 'Train')
make_dir(save_path_train)
save_path_test = os.path.join(Now_Save_Path, 'Test')
make_dir(save_path_test)

all_IoU_OD_list = []
all_Dice_OD_list = []
all_IoU_OC_list = []
all_Dice_OC_list = []

for ratio in RATIO_LIST:
      print("The ratio of training set and testing set is %s:%s." % (ratio[0], ratio[1]))
      ratio_path = os.path.join(DATA_PATH, ratio)

      save_train_ratio_path = os.path.join(save_path_train, ratio)
      make_dir(save_train_ratio_path)
      save_test_ratio_path = os.path.join(save_path_test, ratio)
      make_dir(save_test_ratio_path)

      for experiment in range(EXPERIMENT_NUM):
            if MODEL_TYPE == 'UNet':
                  model = UNet(in_channels=INPUT_CHANNEL, num_classes=NUM_CLASSES, base_c=BASIC_CHANNEL).to(DEVICE)
            elif MODEL_TYPE == 'Swin_Unet':
                  model = SwinUnet(num_classes=NUM_CLASSES, IMG_SIZE=IMAGE_SIZE).to(DEVICE)
            elif MODEL_TYPE == 'Nat_Unet_Mizuno':
                  model = NAT_Unet_Mizuno(in_channels=INPUT_CHANNEL, num_classes=NUM_CLASSES).to(DEVICE)
            elif MODEL_TYPE == 'Nat_Unet_Xue':
                  model = NAT_Unet_Xue(in_channels=INPUT_CHANNEL, num_classes=NUM_CLASSES, base_c=BASIC_CHANNEL).to(DEVICE)
            elif MODEL_TYPE == 'Nat_Multi_Scale_Unet':
                  model = NAT_Multi_Scale_UNET(in_channels=INPUT_CHANNEL, num_classes=NUM_CLASSES).to(DEVICE)
            elif MODEL_TYPE == 'Nat_skip_unet':
                  model = NAT_Skip_Unet(in_channels=INPUT_CHANNEL, num_classes=NUM_CLASSES).to(DEVICE)
            elif MODEL_TYPE == 'Multi_Scale_Output_NAT':
                  model = MSO_NAT(in_channels=INPUT_CHANNEL, num_classes=NUM_CLASSES).to(DEVICE)
            else:
                  raise ValueError('There is no this type of model!')
            print("The No.%d experiment:" % experiment)
            experiment_name = 'experiment_' + str(experiment)

            save_train_experiment_path = os.path.join(save_train_ratio_path, experiment_name)
            make_dir(save_train_experiment_path)
            save_test_experiment_path = os.path.join(save_test_ratio_path, experiment_name)
            make_dir(save_test_experiment_path)

            now_data_path = DATA_PATH + ratio + '/' + experiment_name
            train_image_path_list, train_gt_path_list, test_image_path_list, test_gt_path_list = generate_dataset(now_data_path, GT_TYPE)

            time.sleep(2)

            save_parameter_path = train_process(model, train_image_path_list, train_gt_path_list, save_train_experiment_path,
                                                EPOCHS, LR, GT_TYPE, BATCH_SIZE, NUM_CLASSES, DEVICE)

            mean_IoU_OD, mean_IoU_OC, mean_Dice_OD, mean_Dice_OC = test_process(model, test_image_path_list, test_gt_path_list,
                                                                                save_parameter_path, DEVICE,
                                                                                save_test_experiment_path, GT_TYPE)
            all_IoU_OD_list.append(mean_IoU_OD)
            all_Dice_OD_list.append(mean_Dice_OD)
            all_IoU_OC_list.append(mean_IoU_OC)
            all_Dice_OC_list.append(mean_Dice_OC)
            save_txt(save_test_experiment_path, mean_IoU_OD, mean_IoU_OC, mean_Dice_OD, mean_Dice_OC, EPOCHS,
                     BATCH_SIZE, LR, IMAGE_SIZE, BASIC_CHANNEL, MODEL_TYPE, RATIO_LIST, EXPERIMENT_NUM,
                     GT_TYPE, NUM_CLASSES, INPUT_CHANNEL)

      save_all_txt(save_test_ratio_path, all_IoU_OD_list, all_Dice_OD_list, all_IoU_OC_list, all_Dice_OC_list)

      print('-'*50)