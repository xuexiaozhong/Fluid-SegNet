import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import gc

from DataSet.OCT_DATA import OCTDATA_test
from tools import image_transform, image_transform_3d, cal_metrics, save_image, roi_detection_process

@torch.no_grad()
def test_process(model, pth_path, test_image_path_list, gt_type, device, save_path, using_roi):
    TEST_DATA = OCTDATA_test(test_image_path_list, gt_type, image_transform, image_transform_3d)
    test_loader = DataLoader(dataset=TEST_DATA, batch_size=1, shuffle=False)

    msg = model.load_state_dict(torch.load(pth_path))
    model.eval()

    IoU_list = []
    Dice_list = []

    for scan, scan_3d, label, name, scan_roi, scan_3d_roi in tqdm.tqdm(test_loader):
        scan, scan_3d, scan_roi, scan_3d_roi = scan.to(device), scan_3d.to(device), scan_roi.to(device), scan_3d_roi.to(device)
        label = label.squeeze().detach().numpy()

        if using_roi:
            map1, map2, map3, map4, output = model(scan_roi, scan_3d_roi)
        else:
            map1, map2, map3, map4, output = model(scan, scan_3d)
        out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

        IoU, Dice = cal_metrics(out, label)
        IoU_list.append(IoU)
        Dice_list.append(Dice)

        save_image(save_path, scan.squeeze().cpu().detach().numpy(), out, label, IoU, name)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return np.mean(np.array(IoU_list)), np.mean(np.array(Dice_list))