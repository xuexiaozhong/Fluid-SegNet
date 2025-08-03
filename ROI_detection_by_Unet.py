import cv2
import tqdm

from tools import roi_detection_process, make_dir
import os
import matplotlib.pyplot as plt

data_path = './UMN_data'
data_type = ['AMD', 'DME']

for type in data_type:
    type_path = os.path.join(data_path, type)
    for subject in tqdm.tqdm(os.listdir(type_path)):
        subject_path = os.path.join(type_path, subject)
        save_roi_path = os.path.join(subject_path, 'roi')
        save_neighbor_roi_path = os.path.join(subject_path, 'neighbor_roi')
        make_dir(save_roi_path)
        make_dir(save_neighbor_roi_path)
        for mode in os.listdir(subject_path):
            mode_path = os.path.join(subject_path, mode)
            if mode == 'data':
                for image_name in os.listdir(mode_path):
                    image_path = os.path.join(mode_path, image_name)
                    now_save_roi_path = os.path.join(save_roi_path, image_name)
                    roi_image = roi_detection_process(image_path)
                    cv2.imwrite(now_save_roi_path, roi_image)
            elif mode == 'neighbor':
                for neighbor_name in os.listdir(mode_path):
                    neighbor_path = os.path.join(mode_path, neighbor_name)
                    now_save_neighbor_roi_path = os.path.join(save_neighbor_roi_path, neighbor_name)
                    neighbor_roi = roi_detection_process(neighbor_path)
                    cv2.imwrite(now_save_neighbor_roi_path, neighbor_roi)
            else:
                continue