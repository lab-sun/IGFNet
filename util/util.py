# By Yuxiang Sun, Dec. 4, 2020
# Email: sun.yuxiang@outlook.com
import os, shutil, stat
import numpy as np 
from PIL import Image 
import torch
 
# 0:'unlabelled', 1:'person', 2:'car', 3:'bike', 4:'dog', 5:'car_stop', 6:'curb', 7:'yellow_ballard', 8:'trash_can', 9:'parking_barrier'
def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette

def visualize(image_name, predictions, vis_dir):
    visualize_dir = vis_dir
    palette = get_palette()
    pred = predictions.cpu().numpy()
    img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
        img[pred == cid] = palette[cid]
    img = Image.fromarray(np.uint8(img))
    # image_name = str(image_name)
    # torch.save(img,os.path.join(visualize_dir, image_name + '.png'))
    img.save(os.path.join(visualize_dir, image_name[0] + '.png'), format='png')

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    F1_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN
        if (recall_per_class[cid] == np.nan) | (precision_per_class[cid] == np.nan) |(precision_per_class[cid]==0)|(recall_per_class[cid]==0):
            F1_per_class[cid] = np.nan
        else :
            F1_per_class[cid] = 2 / (1/precision_per_class[cid] +1/recall_per_class[cid])

    return precision_per_class, recall_per_class, iou_per_class,F1_per_class

def label2onehot(label,n_class,input_w,input_h):
    label = torch.tensor(label).unsqueeze(0)
    print(label.size())
    onehot = torch.zeros(n_class, input_h, input_w).long()   # 先生成模板
    print(onehot.size())
    onehot.scatter_(0, label, 1).float()   # 这个就是生成6个channel的， scatter_这个函数不必理解太深，知道这么一个用法就OK了

    return onehot

    