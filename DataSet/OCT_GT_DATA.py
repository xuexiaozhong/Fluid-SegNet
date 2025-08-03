from torch.utils.data import Dataset
import numpy as np
import cv2

class OCTGTDATA(Dataset):
    def __init__(self, train_scan_list, gt_type, intermittent_list=None, transform_output=None, transform_input=None, generating=False):
        self.train_scan_list = train_scan_list
        self.GT_type = gt_type
        self.transform_output = transform_output
        self.transform_input = transform_input
        self.intermittent_list = intermittent_list
        self.generating = generating


    def __len__(self):
        self.length = len(self.train_scan_list)
        return self.length

    def __getitem__(self, item):
        scan_path = self.train_scan_list[item]
        scan = cv2.imread(scan_path, 0)
        scan_path_split = scan_path.split('\\')
        name = scan_path_split[4]
        gt1_path = scan_path_split[0]+'\\'+scan_path_split[1]+'\\'+scan_path_split[2]+'\\gt1\\'+scan_path_split[4]
        gt2_path = scan_path_split[0]+'\\'+scan_path_split[1]+'\\'+scan_path_split[2]+'\\gt2\\'+scan_path_split[4]
        roi_path = scan_path_split[0]+'\\'+scan_path_split[1]+'\\'+scan_path_split[2]+'\\roi\\'+scan_path_split[4]
        scan_roi = cv2.imread(roi_path, 0)

        gt1 = cv2.imread(gt1_path, 0)
        gt2 = cv2.imread(gt2_path, 0)

        gt = np.array(gt1, dtype=int) + np.array(gt2, dtype=int)
        gt[gt == 255] = 0
        gt[gt == 510] = 255

        if self.GT_type == 'gt1':
            gt1_tensor = self.transform_output(gt1)
            if self.generating:
                input_tensor = self.transform_input(scan, gt1, 1)
                roi_tensor = self.transform_input(scan_roi, gt1, 1)
            else:
                intermittent = self.intermittent_list[item]
                input_tensor = self.transform_input(scan, gt1, intermittent)
                roi_tensor = self.transform_input(scan_roi, gt1, intermittent)
            return input_tensor, gt1_tensor, name, roi_tensor
        elif self.GT_type == 'gt2':
            gt2_tensor = self.transform_output(gt2)
            if self.generating:
                input_tensor = self.transform_input(scan, gt2, 1)
                roi_tensor = self.transform_input(scan_roi, gt2, 1)
            else:
                intermittent = self.intermittent_list[item]
                input_tensor = self.transform_input(scan, gt2, intermittent)
                roi_tensor = self.transform_input(scan_roi, gt2, intermittent)
            return input_tensor, gt2_tensor, name, roi_tensor
        elif self.GT_type == 'both':
            gt_tensor = self.transform_output(gt)
            if self.generating:
                input_tensor = self.transform_input(scan, gt, 1)
                roi_tensor = self.transform_input(scan_roi, gt, 1)
            else:
                intermittent = self.intermittent_list[item]
                input_tensor = self.transform_input(scan, gt, intermittent)
                roi_tensor = self.transform_input(scan_roi, gt, intermittent)
            return input_tensor, gt_tensor, name, roi_tensor
        else:
            raise ValueError('The GT_type should be chosen from gt1, gt2, and both!')