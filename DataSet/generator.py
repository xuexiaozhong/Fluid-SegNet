import tqdm
import os
import random

def generate_dataset(path):
    AMD_path = os.path.join(path, 'AMD')
    DME_path = os.path.join(path, 'DME')

    shuffled_AMD = random.sample(os.listdir(AMD_path), len(os.listdir(AMD_path)))
    shuffled_DME = random.sample(os.listdir(DME_path), len(os.listdir(DME_path)))

    AMD_project_list = []
    DME_project_list = []

    for i in shuffled_AMD:
        AMD_project_list.append(os.path.join(AMD_path, i))
    for j in shuffled_DME:
        DME_project_list.append(os.path.join(DME_path, j))

    return shuffled_AMD, shuffled_DME, AMD_project_list, DME_project_list