
from utils import load_save_pkl

train_sdp_dep_data = load_save_pkl.load_pickle_file('/home/ubuntu/subbu/cross_sentence_thesis/data/processed_input/BB2016/train/k_0/BB2016_train_data_w_others_k_0_sdp_dep.pkl')


print('sentence:', train_sdp_dep_data[0][0])
print
print
#print('\n\n parse tree:', train_sdp_dep_data[0][1])
print
print
#print('\n\n parse tree first sentence:', train_sdp_dep_data[0][1][0][0])

len_parse_tree=len( train_sdp_dep_data[0][1][0][0])

#print(len_parse_tree)

for i in range(len_parse_tree):
	#for j in range(4):
		#print(train_sdp_dep_data[0][1][0][0][i][j])
	if i!=train_sdp_dep_data[0][1][0][0][i][0]:
		train_sdp_dep_data[0][1][0][0][i][0]=train_sdp_dep_data[0][1][0][0][i][0]-1
		train_sdp_dep_data[0][1][0][0][i][3] = train_sdp_dep_data[0][1][0][0][i][3] - 1


print(train_sdp_dep_data[0][1][0][0][0])
print 
print
#print('\n\n sdp:', train_sdp_dep_data[0][1][1])



exit()
dev_sdp_dep_data = load_save_pkl.load_pickle_file('/home/ubuntu/subbu/cross_sentence_thesis/data/processed_input/BB2016/train/k_0/BB2016_dev_data_w_others_k_0_sdp_dep.pkl')








test_sdp_dep_data = load_save_pkl.load_pickle_file('/home/ubuntu/subbu/cross_sentence_thesis/data/processed_input/BB2016/train/k_0/BB2016_test_data_w_others_k_0_sdp_dep.pkl')
