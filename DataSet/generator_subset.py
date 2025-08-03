import os
import random

def generate_subset_dataset(shuffled_AMD, shuffled_DME, AMD_subset_len, DME_subset_len, test_subset_num):
    if len(AMD_subset_len) != len(DME_subset_len):
        raise ValueError('The length of AMD_subset_len and DME_subset_len should be same!')

    train_project_list = []
    test_project_list = []

    shuffled_AMD_copy = shuffled_AMD.copy()
    shuffled_DME_copy = shuffled_DME.copy()

    for index, i in enumerate(AMD_subset_len):
        if index == test_subset_num:
            for j in range(i):
                test_project_list.append(shuffled_AMD_copy[0])
                del shuffled_AMD_copy[0]
        else:
            for k in range(i):
                train_project_list.append(shuffled_AMD_copy[0])
                del shuffled_AMD_copy[0]

    for index, i in enumerate(DME_subset_len):
        if index == test_subset_num:
            for j in range(i):
                test_project_list.append(shuffled_DME_copy[0])
                del shuffled_DME_copy[0]
        else:
            for k in range(i):
                train_project_list.append(shuffled_DME_copy[0])
                del shuffled_DME_copy[0]

    return train_project_list, test_project_list