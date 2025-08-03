'''
The variables for this project
'''

'''
Hyper parameters
'''
TRAIN_DECODER_EPOCH = 5
TRAIN_EACH_MAP_EPOCH = 10
TRAIN_FINAL_EPOCH = 1
LR = 1e-3
BATCH_SIZE = 4
BASIC_CHANNEL = 32
NUM_CLASSES = 2
USING_MULTI_LOSS = False
USING_INTERMITTENT_MAP = False
WITH_GT_GROUP = 1 #TRAIN_DECODER_EPOCH // 5
USING_MAX_MAP = False
GT_TRAIN_GT = False
GT_MAP_NORMALIZATION = False
USING_ROI_DETECTION = False
# USING_DILATED_CONV = False
MODEL_TYPE = 'Y-Net'   # Y-Net, W-Net, Dilated_YNet, U_Net, U_Net_pp, CBAM_Dilated_YNet

'''
Global variables
'''
# IMAGE_SIZE = [1024, 496]
IMAGE_SIZE = [1024, 512] if MODEL_TYPE=='W-Net' else [1024, 496]
DEVICE = 'cuda'
GT_TYPE = 'gt1' # gt1, gt2, both
DATA_PATH = './UMN_data'
AMD_SUBSET_LEN = [5, 5, 5, 5, 4]
DME_SUBSET_LEN = [6, 6, 6, 6, 5]
FOLD_NUM = len(AMD_SUBSET_LEN)
SAVE_PATH = './output'