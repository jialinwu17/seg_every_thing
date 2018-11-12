import os,sys, cPickle
test_list = cPickle.load(open('test_list.pkl'))
from tqdm import tqdm
for f in tqdm(test_list):
	imgid = int(f.split('_')[-1].split('.')[0])
	npz_name = '%d.npz'%imgid
	os.system('mv npz_features/coco_train2014/%s npz_features/coco_test2015/%s'%(npz_name, npz_name))
