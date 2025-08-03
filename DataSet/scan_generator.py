import os
import random
import time
import tqdm


def generate_scan_dataset(train_project_path_list, test_project_path_list):
    train_scan_path_list = []
    test_scan_path_list = []

    print('Loading the training scan:')
    time.sleep(0.5)
    for i in tqdm.tqdm(train_project_path_list):
        train_data_path = os.path.join(i, 'data')
        for j in os.listdir(train_data_path):
            train_scan_path_list.append(os.path.join(train_data_path, j))

    print('Loading the testing scan')
    time.sleep(0.5)
    for k in tqdm.tqdm(test_project_path_list):
        test_data_path = os.path.join(k, 'data')
        for l in os.listdir(test_data_path):
            test_scan_path_list.append(os.path.join(test_data_path, l))

    train_scan_path_list = random.sample(train_scan_path_list, len(train_scan_path_list))
    test_scan_path_list = random.sample(test_scan_path_list, len(test_scan_path_list))

    print("Length of training scan is: ", len(train_scan_path_list))
    time.sleep(0.5)
    print("Length of testing scan is: ", len(test_scan_path_list))
    time.sleep(0.5)

    return train_scan_path_list, test_scan_path_list