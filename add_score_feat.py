import numpy as np
from os import listdir
from tqdm import tqdm
'''
train_folder = 'npz_features/coco_train2014'
file_list = listdir(train_folder)
for fidx in tqdm(xrange(len(file_list))):
    f = file_list[fidx]
    if f[:5] == 'score':
        continue
    npz_name = train_folder + '/' + f
    npz_save_name = train_folder + '/score_' + f 
    feats = np.load(npz_name)
    roi_feat = feats['roi_feat']
    cls_segms = feats['cls_segms']
    n_ins = roi_feat.shape[0]
    score_feat = np.zeros((n_ins))
    count = 0
    for i in xrange( 1, 3002):
        for j in xrange(len(cls_segms[i])):
            score_feat[count] = i
            count += 1
    assert count == n_ins
    np.savez(npz_save_name, score_feat = score_feat)
        
train_folder = 'npz_features/coco_val2014'
file_list = listdir(train_folder)
for fidx in tqdm(xrange(len(file_list))):
    f = file_list[fidx]
    npz_name = train_folder + '/' + f
    npz_save_name = train_folder + '/score_' + f
    feats = np.load(npz_name)
    roi_feat = feats['roi_feat']
    cls_segms = feats['cls_segms']
    n_ins = roi_feat.shape[0]
    score_feat = np.zeros((n_ins))
    count = 0
    for i in xrange( 1, 3002):
        for j in xrange(len(cls_segms[i])):
            score_feat[count] = i
            count += 1
    assert count == n_ins
    np.savez(npz_save_name, score_feat = score_feat)
'''

train_folder = 'npz_features/coco_test2015'
file_list = listdir(train_folder)
for fidx in tqdm(xrange(len(file_list))):
    f = file_list[fidx]
    npz_name = train_folder + '/' + f
    npz_save_name = train_folder + '/score_' + f
    feats = np.load(npz_name)
    roi_feat = feats['roi_feat']
    cls_segms = feats['cls_segms']
    n_ins = roi_feat.shape[0]
    score_feat = np.zeros((n_ins))
    count = 0
    for i in xrange( 1, 3002):
        for j in xrange(len(cls_segms[i])):
            score_feat[count] = i
            count += 1
    assert count == n_ins
    np.savez(npz_save_name, score_feat = score_feat)