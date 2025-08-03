from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
import os

class OCTDATA(Dataset):
    def __init__(self, scan_path_list, map_path, gt_type, index, transform=None, transform_3d=None, normalization=True):
        self.scan_path_list = scan_path_list
        self.map_path = map_path
        self.gt_type = gt_type
        self.index = index
        self.transform = transform
        self.transform_3d = transform_3d
        self.normalization = normalization


    def __len__(self):
        self.length = len(self.scan_path_list)
        return self.length

    def __getitem__(self, item):
        scan_path = self.scan_path_list[item]
        scan_path_split = scan_path.split('\\')
        name = scan_path_split[4]
        gt1_path = scan_path_split[0] + '\\' + scan_path_split[1] + '\\' + scan_path_split[2] + '\\gt1\\' + \
                   scan_path_split[4]
        gt2_path = scan_path_split[0] + '\\' + scan_path_split[1] + '\\' + scan_path_split[2] + '\\gt2\\' + \
                   scan_path_split[4]
        roi_path = scan_path_split[0] + '\\' + scan_path_split[1] + '\\' + scan_path_split[2] + '\\roi\\' + \
                   scan_path_split[4]
        neighbor_path = scan_path_split[0] + '\\' + scan_path_split[1] + '\\' + scan_path_split[2] + '\\neighbor\\'
        neighbor_roi_path = scan_path_split[0] + '\\' + scan_path_split[1] + '\\' + scan_path_split[2] + '\\neighbor_roi\\'


        name_split = name.split('_')
        scan_num = int(name_split[-1].split('s')[-1].split('.')[0])
        before_num = scan_num - 1
        after_num = scan_num + 1
        before_name = name_split[0] + '_' + name_split[1] + '_' + name_split[2] + '_s' + str(before_num) + '.png'
        after_name = name_split[0] + '_' + name_split[1] + '_' + name_split[2] + '_s' + str(after_num) + '.png'
        before_scan_path = neighbor_path + before_name
        after_scan_path = neighbor_path + after_name
        before_scan_roi_path = neighbor_roi_path + before_name
        after_scan_roi_path = neighbor_roi_path + after_name

        scan = cv2.imread(scan_path, 0)
        scan_roi = cv2.imread(roi_path, 0)
        gt1 = cv2.imread(gt1_path, 0)
        gt2 = cv2.imread(gt2_path, 0)
        gt = np.array(gt1, dtype=int) + np.array(gt2, dtype=int)
        gt[gt == 255] = 0
        gt[gt == 510] = 255
        before_scan = cv2.imread(before_scan_path, 0)
        after_scan = cv2.imread(after_scan_path, 0)
        before_scan_roi = cv2.imread(before_scan_roi_path, 0)
        after_scan_roi = cv2.imread(after_scan_roi_path, 0)

        if self.index < 4:
            decoder_name = name.split('.')[0] + '_decoder' + str(self.index+1) + '.npy'
            decoder_path = os.path.join(self.map_path, decoder_name)
            decoder_npy = np.load(decoder_path)
            if self.normalization:
                decoder_npy = (decoder_npy - np.min(decoder_npy)) / (np.max(decoder_npy) - np.min(decoder_npy))
            decoder = torch.tensor(decoder_npy)
        else:
            decoder = torch.tensor(np.array([1]))

        scan_tensor = self.transform(scan)
        scan_3d_tensor = self.transform_3d(scan, before_scan, after_scan)
        scan_roi_tensor = self.transform(scan_roi)
        scan_3d_roi_tensor = self.transform_3d(scan_roi, before_scan_roi, after_scan_roi)

        if self.gt_type == 'gt1':
            gt1_tensor = self.transform(gt1)
            return scan_tensor, scan_3d_tensor, gt1_tensor, name, decoder, scan_roi_tensor, scan_3d_roi_tensor
        elif self.gt_type == 'gt2':
            gt2_tensor = self.transform(gt2)
            return scan_tensor, scan_3d_tensor, gt2_tensor, name, decoder, scan_roi_tensor, scan_3d_roi_tensor
        elif self.gt_type == 'both':
            gt_tensor = self.transform(gt)
            return scan_tensor, scan_3d_tensor, gt_tensor, name, decoder, scan_roi_tensor, scan_3d_roi_tensor
        else:
            raise ValueError('The GT_type should be chosen from gt1, gt2, and both!')

class OCTDATA_test(Dataset):
    def __init__(self, scan_path_list, gt_type, transform=None, transform_3d=None):
        self.scan_path_list = scan_path_list
        self.gt_type = gt_type
        self.transform = transform
        self.transform_3d = transform_3d


    def __len__(self):
        self.length = len(self.scan_path_list)
        return self.length

    def __getitem__(self, item):
        scan_path = self.scan_path_list[item]
        scan_path_split = scan_path.split('\\')
        name = scan_path_split[4]
        gt1_path = scan_path_split[0] + '\\' + scan_path_split[1] + '\\' + scan_path_split[2] + '\\gt1\\' + \
                   scan_path_split[4]
        gt2_path = scan_path_split[0] + '\\' + scan_path_split[1] + '\\' + scan_path_split[2] + '\\gt2\\' + \
                   scan_path_split[4]
        roi_path = scan_path_split[0] + '\\' + scan_path_split[1] + '\\' + scan_path_split[2] + '\\roi\\' + \
                   scan_path_split[4]
        neighbor_path = scan_path_split[0] + '\\' + scan_path_split[1] + '\\' + scan_path_split[2] + '\\neighbor\\'
        neighbor_roi_path = scan_path_split[0] + '\\' + scan_path_split[1] + '\\' + scan_path_split[2] + '\\neighbor_roi\\'

        name_split = name.split('_')
        scan_num = int(name_split[-1].split('s')[-1].split('.')[0])
        before_num = scan_num - 1
        after_num = scan_num + 1
        before_name = name_split[0] + '_' + name_split[1] + '_' + name_split[2] + '_s' + str(before_num) + '.png'
        after_name = name_split[0] + '_' + name_split[1] + '_' + name_split[2] + '_s' + str(after_num) + '.png'
        before_scan_path = neighbor_path + before_name
        after_scan_path = neighbor_path + after_name
        before_scan_roi_path = neighbor_roi_path + before_name
        after_scan_roi_path = neighbor_roi_path + after_name

        scan = cv2.imread(scan_path, 0)
        scan_roi = cv2.imread(roi_path, 0)
        gt1 = cv2.imread(gt1_path, 0)
        gt2 = cv2.imread(gt2_path, 0)
        gt = np.array(gt1, dtype=int) + np.array(gt2, dtype=int)
        gt[gt == 255] = 0
        gt[gt == 510] = 255
        before_scan = cv2.imread(before_scan_path, 0)
        after_scan = cv2.imread(after_scan_path, 0)
        before_scan_roi = cv2.imread(before_scan_roi_path, 0)
        after_scan_roi = cv2.imread(after_scan_roi_path, 0)

        scan_tensor = self.transform(scan)
        scan_roi_tensor = self.transform(scan_roi)
        scan_3d_tensor = self.transform_3d(scan, before_scan, after_scan)
        scan_3d_roi_tensor = self.transform_3d(scan_roi, before_scan_roi, after_scan_roi)

        if self.gt_type == 'gt1':
            gt1_tensor = self.transform(gt1)
            return scan_tensor, scan_3d_tensor, gt1_tensor, name, scan_roi_tensor, scan_3d_roi_tensor
        elif self.gt_type == 'gt2':
            gt2_tensor = self.transform(gt2)
            return scan_tensor, scan_3d_tensor, gt2_tensor, name, scan_roi_tensor, scan_3d_roi_tensor
        elif self.gt_type == 'both':
            gt_tensor = self.transform(gt)
            return scan_tensor, scan_3d_tensor, gt_tensor, name, scan_roi_tensor, scan_3d_roi_tensor
        else:
            raise ValueError('The GT_type should be chosen from gt1, gt2, and both!')