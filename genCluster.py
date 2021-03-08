import argparse
import coloredlogs, logging
import logging.config
import numpy as np
import os
import sys

from config import *
from data.datagenerator import DataGenerator 
import pdb
NUM_CLUSTERS = 512
UPRIGHT_AXIS = 2  # Will learn invariance along this axis
VAL_PROPORTION = 1.0

# Arguments
parser = argparse.ArgumentParser(description='Trains pointnet')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use (default: 0)')
# data
parser.add_argument('--data_dim', type=int, default=6,
                    help='Input dimension for data. Note: Feat3D-Net will only use the first 3 dimensions (default: 6)')
parser.add_argument('--data_dir', type=str, default='../data/oxford',
                    help='Path to dataset. Should contain "train" and "clusters" folders')
parser.add_argument('--recall', type=int, default=0,
        help='gen or recall: 1 is recall, 0 is gen') 
args = parser.parse_args()

def load_validation_groundtruths(fname, proportion=1):
    groundtruths = []
    iGt = 0
    with open(fname) as fid:
        fid.readline()
        for line in fid:
            groundtruths.append((iGt, int(line.split()[-1])))
            iGt += 1

    if 0 < proportion < 1:
        skip = int(1.0/proportion)
        groundtruths = groundtruths[0::skip]

    return groundtruths


def gen_cluster_point_cloud_and_kp(val_folder, val_groundtruths, data_dim):
    os.makedirs(val_folder+'_pckp', exist_ok=True)  
    os.makedirs(val_folder+'_pckp/pc', exist_ok=True)
    os.makedirs(val_folder+'_pckp/kp', exist_ok=True) 

    if val_groundtruths is None or len(val_groundtruths) == 0:
        return 1

    positive_dist = []
    negative_dist = []

    for iTest in range(0, len(val_groundtruths), NUM_CLUSTERS):

        clouds1, clouds2 = [], []
        # We batch the validation by stacking all the validation clusters into a single point cloud,
        # while keeping them apart such that they do not overlap each other. This way NUM_CLUSTERS
        # clusters can be computed in a single pass
        for jTest in range(iTest, min(iTest + NUM_CLUSTERS, len(val_groundtruths))):
            offset = (jTest - iTest) * 100
            cluster_idx = val_groundtruths[jTest][0]

            cloud1 = DataGenerator.load_point_cloud(
                os.path.join(val_folder, '{}_0.bin'.format(cluster_idx)), data_dim)
            cloud1[:, 0] += offset
            clouds1.append(cloud1)

            cloud2 = DataGenerator.load_point_cloud(
                os.path.join(val_folder, '{}_1.bin'.format(cluster_idx)), data_dim)
            cloud2[:, 0] += offset
            clouds2.append(cloud2)

        offsets = np.arange(0, NUM_CLUSTERS * 100, 100)
        num_clusters = min(len(val_groundtruths) - iTest, NUM_CLUSTERS)
        offsets[num_clusters:] = 0
        offsets = np.pad(offsets[:, None], ((0, 0), (0, 2)), mode='constant', constant_values=0)

        clouds1 = np.concatenate(clouds1, axis=0)
        clouds2 = np.concatenate(clouds2, axis=0)

        with open(os.path.join(val_folder+'_pckp/pc', '{}_0.bin'.format(iTest)),'wb') as f:
            clouds1.tofile(f)
        with open(os.path.join(val_folder+'_pckp/pc', '{}_1.bin'.format(iTest)),'wb') as f:
            clouds2.tofile(f)                                             

        with open(os.path.join(val_folder+'_pckp/kp', '{}_kp.bin'.format(iTest)),'wb') as f:
            offsets.tofile(f)                                             
 
def recall(val_folder, val_groundtruths, data_dim):
    if val_groundtruths is None or len(val_groundtruths) == 0:
        return 1
    positive_dist = []
    negative_dist = []
     

    for iTest in range(0, len(val_groundtruths), NUM_CLUSTERS):
        num_clusters = min(len(val_groundtruths) - iTest, NUM_CLUSTERS) 
        desc1 = DataGenerator.load_point_cloud(
                os.path.join(val_folder+'_pckp/desc', '{}_0.bin'.format(iTest)), data_dim)
        desc1 = desc1[:,3:]

        desc2 = DataGenerator.load_point_cloud(
                os.path.join(val_folder+'_pckp/desc', '{}_1.bin'.format(iTest)), data_dim)
        desc2 = desc2[:,3:]

        d = np.sqrt(np.sum(np.square(np.squeeze(desc1 - desc2)), axis=1))
        d = d[:num_clusters]

        positive_dist += [d[i] for i in range(len(d)) if val_groundtruths[iTest + i][1] == 1]
        negative_dist += [d[i] for i in range(len(d)) if val_groundtruths[iTest + i][1] == 0]
 
    d_at_95_recall = np.percentile(positive_dist, 95)
    num_FP = np.count_nonzero(np.array(negative_dist) < d_at_95_recall)
    num_TN = len(negative_dist) - num_FP
    fp_rate = num_FP / (num_FP + num_TN)
    return fp_rate
        

if __name__ == '__main__':
    val_folder = os.path.join(args.data_dir, 'clusters')
    val_groundtruths = load_validation_groundtruths(os.path.join(val_folder, 'filenames.txt'), proportion=VAL_PROPORTION)
    if args.recall == 0:
        gen_cluster_point_cloud_and_kp(val_folder, val_groundtruths, args.data_dim)
    else:
        fp = recall(val_folder, val_groundtruths, 33+3)
        print('fp is', fp)
     
