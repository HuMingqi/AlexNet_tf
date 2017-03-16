'''
Author: hiocde
Email: hiocde@gmail.com
Start: 3.13.17
Completion: 3.16.17
Original: Object retrieve on Paris Dataset(VGG) using feature extracted by play.py.
Domain: CBIR, Object Retrieve.
'''

import json
import heapq
import os
import sys
# import datetime
import numpy as np

def main(argv):
	'''
	Query and compute PR curve.
	Args:
		1: query set path(one of {fc1,fc2,fc3})
		2: feature lib path(one of {fc1,fc2,fc3},corresponding to 1.)
		3: ground truth dir;
		4: output dir including similar images and PR file.
	'''	
	query_set=argv[1]; feature_lib=argv[2]; gtd=argv[3]; output_dir=argv[4]
	
	res= query(query_set, feature_lib, 100)
	print('******Query all ok!******')
	name_queryid=name_queryid_map(gtd)
	similar_f= output_dir+'/similar_based_'+query_set.split('/')[-1].split('_')[0]
	with open(similar_f, 'w') as sf:
		sf.write(json.dumps(dict(res)))
		print('Write result ok!')
	pr_f= output_dir+'/PR_based_'+query_set.split('/')[-1].split('_')[0]
	with open(pr_f,'a') as prf:
		for re in res:
			qname= re[0].split('\\')[-1][:-4]
			query_id= name_queryid[qname]	# exp: defense_1			
			gtp_good= gtd+'/'+query_id+'_good.txt'
			gtp_ok= gtd+'/'+query_id+'_ok.txt'
			gtp_junk= gtd+'/'+query_id+'_junk.txt'
			pos_set= read_list(gtp_good)+read_list(gtp_ok)
			ignore_set= read_list(gtp_junk)	# junk ground truth is ambiguous, ignored
			
			precision, recall= compute_pr([sp.split('\\')[-1][:-4] for sp in re[1]], pos_set, ignore_set)
			prf.write(query_id+'_query\n')
			prf.write(' '.join(map(str,precision))+'\n')
			prf.write(' '.join(map(str,recall))+'\n')
	print("Write PR ok!")


def query(query_set, feature_lib, rank):
	'''
	Args:
		query_set: path to query set feature file, '{query_img_name:feature, ...}';
		feature_lib: path to features lib file;
		rank: top k similar images.
	Return:
		'[(query_img_name, [similar_img_name, ...]), ...]'
	'''
	with open(query_set) as qs, open(feature_lib) as f:
		query_set_dict= json.loads(qs.read())
		feature_lib_dict= json.loads(f.read())
	li= []
	for pair in query_set_dict.items():
		li.append(once_query(pair, feature_lib_dict, rank))
	return li

def once_query(query_feature, feature_lib_dict, rank):
	'''
	Args:
		query_feature: a tuple like '(query_img_name, feature)';
		feature_lib_dict: features lib dict;
		rank: top k similar images.
	Return:
		query_img_name, top-k similar images name list.
	'''
	# with open(feature_lib) as f:
	# 	feature_lib_dict= json.loads(f.read())
	dists= dist(query_feature[1], feature_lib_dict)
	top_k= heapq.nsmallest(rank, dists, key= lambda d:d[1])
	print(query_feature[0]+' query completed.')
	return query_feature[0], list(zip(*top_k))[0]

def dist(query_feature, feature_lib_dict):
	dists = []	
	for image_feature in list(feature_lib_dict.values()):
		# t0= datetime.datetime.now()
		# dist = 0
		# for feature_x, feature_y in zip(query_feature, image_feature):
		# 	dist = dist + (feature_x-feature_y)**2
		dist= np.linalg.norm(np.array(query_feature)-np.array(image_feature))
		# t1= datetime.datetime.now()
		# print(t1-t0)
		dists.append(dist)
	return list(zip(list(feature_lib_dict.keys()), dists))

def read_list(gtp):
	with open(gtp) as gt:		
		# return gt.readlines()	# '\n' included
		return gt.read().split('\n')[:-1]

def compute_pr(ranked_list, pos_list, ignore_list):	
	pos_num= len(pos_list)
	rank= 0
	intersection= 0
	precision=[]; recall=[]
	for pred in ranked_list:		
		if pred in ignore_list:
			precision.append(precision[-1])	# stay unchanged
			recall.append(recall[-1])
			continue
		if pred in pos_list:
			intersection+=1
		
		rank+=1
		precision.append(intersection/rank)
		recall.append(intersection/pos_num)

	return precision, recall

def name_queryid_map(gtd):
	name_queryid={}
	for fn in os.listdir(gtd):
		if fn.endswith('query.txt'):
			with open(os.path.join(gtd,fn)) as qf:
				qname= qf.read().split(' ')[0]	# exp: paris_defense_000605
				name_queryid[qname]= fn[:-10]
	return name_queryid


if __name__ == '__main__':	
	main(sys.argv)