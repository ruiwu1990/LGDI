import matplotlib.pyplot as plt
import pandas as pd
import sys
import csv
import math
import neat
from sklearn.metrics import accuracy_score
from dateutil.parser import parse
from datetime import datetime
# from SPOT.spot import bidSPOT,dSPOT,SPOT
from functools import reduce
import numpy as np
import random

def col_based_combine_matrix(df, n_star):
	'''
	new col based reshape operation
	'''
	if df.shape[0]%n_star != 0:
		print('need to trim df!')
		return -1

	loop_count = n_star
	reshape_chunk_row_count = int(df.shape[0]/n_star)
	# df.shape[1]->original col num, n_star-1 because need to remove duplicate col chunk
	reshape_chunk_col_count = int(df.shape[1]*(n_star-1))
	combined_matrix = np.array([])
	
	for i in range(loop_count):
		# remove duplicate rows
		np_array = df.to_numpy()
		# remove duplicate chunk
		duplicate_chunk_start = i*reshape_chunk_row_count
		duplicate_chunk_end = duplicate_chunk_start + reshape_chunk_row_count
		if i == 0:
			# first chunk
			np_array = np_array[duplicate_chunk_end:,]
		elif loop_count-1 == i:
			# last chunk
			np_array = np_array[:duplicate_chunk_start,]
		else:
			upper_matrix = np_array[:duplicate_chunk_start,]
			lower_matrix = np_array = np_array[duplicate_chunk_end:,]
			np_array = np.concatenate((upper_matrix,lower_matrix))

		# reshaped cols will be combined horizontally
		reshape_matrix = np.array([])
		for m in range(loop_count-1):
			if m == 0:
				reshape_matrix = np_array[:reshape_chunk_row_count,]
			# elif m == loop_count-1-1:
			# 	reshape_matrix = np.concatenate([reshape_matrix,np_array[reshape_chunk_row_count*(loop_count-2):,]],axis=1)
			else:
				start = m*reshape_chunk_row_count
				end = start + reshape_chunk_row_count
				reshape_matrix = np.concatenate([reshape_matrix,np_array[start:end,]],axis=1)

		if i == 0:
			combined_matrix = reshape_matrix
		else:
			combined_matrix = np.concatenate((combined_matrix,reshape_matrix))
	
	return combined_matrix

def logsinh(x, tmp_min, logsinh_a, logsinh_b, epsilon):
	'''
	tmp_min is the minimum value of array x
	'''
	return math.log10(math.sinh((x-tmp_min)*logsinh_b+logsinh_a+epsilon))/logsinh_b

def convert_row_index_row_based_reshape(original_extreme_row_index_list, row_based_reshape_factor, total_rows):
	'''
	this function transform original row index to reshape matrix index
	based on the row based reshape method
	row_based_reshape_factor denotes the number of rows will be combined, e.g. originally matrix is n1 * n2 and factor is n`
	the transformed matrix will be (n1/n`)*(n2*n`)
	'''
	tmp_extreme = [int(i/row_based_reshape_factor) for i in original_extreme_row_index_list]
	# remove duplicate element
	reshape_extreme_row_index_list = []
	[reshape_extreme_row_index_list.append(x) for x in tmp_extreme if x not in reshape_extreme_row_index_list]
	# use ceil function because extra rows will be counted as a new row for the reshape matrix
	reshape_normal_row_index_list = np.setdiff1d(range(math.ceil(total_rows/row_based_reshape_factor)),reshape_extreme_row_index_list)
	# sort
	reshape_extreme_row_index_list = sorted(reshape_extreme_row_index_list)
	reshape_normal_row_index_list = sorted(reshape_normal_row_index_list)

	return reshape_normal_row_index_list, reshape_extreme_row_index_list

def convert_row_index_col_based_reshape(original_extreme_row_index_list, col_based_reshape_factor, total_rows):
	'''
	this function transform original row index to reshape matrix index
	based on the col based reshape method
	col_based_reshape_factor denotes the number of cols will be combined horizontally, e.g. originally matrix is n1 * n2 and factor is n`
	the transformed matrix will be n`*(n1*n2/n`)
	'''
	tmp_extreme = [int(i%col_based_reshape_factor) for i in original_extreme_row_index_list]
	# remove duplicate element
	reshape_extreme_row_index_list = []
	[reshape_extreme_row_index_list.append(x) for x in tmp_extreme if x not in reshape_extreme_row_index_list]
	# use ceil function because extra rows will be counted as a new row for the reshape matrix
	reshape_normal_row_index_list = np.setdiff1d(range(math.ceil(col_based_reshape_factor)),reshape_extreme_row_index_list)
	# sort
	reshape_extreme_row_index_list = sorted(reshape_extreme_row_index_list)
	reshape_normal_row_index_list = sorted(reshape_normal_row_index_list)

	return reshape_normal_row_index_list, reshape_extreme_row_index_list

def combine_normal_extreme(total_data_size, results_alarms, results_normal, array_filled_extreme, array_filled_normal, shape_x, shape_y):
	'''
	This function combine normal and extreme events
	shape_x, shape_y are the original df x and y size
	array_filled_extreme, array_filled_normal stores original extreme and normal events
	results_alarms, results_normal stores the mapping index of original matrix
	'''
	array_filled = np.array([])
	tmp_normal_index = 0
	tmp_extreme_index = 0
	for df_index in range(0,total_data_size):
		# current is extreme event
		if df_index in results_alarms:
			array_filled = np.append(array_filled, array_filled_extreme[tmp_extreme_index])
			tmp_extreme_index = tmp_extreme_index + 1
		elif df_index in results_normal:
			array_filled = np.append(array_filled, array_filled_normal[tmp_normal_index])
			tmp_normal_index = tmp_normal_index + 1

	# reshape from 1d to 2d
	return array_filled.reshape(shape_x,shape_y)

def make_holes_matrix(df, percentage, exclude_col, seed = 1):
# def make_holes_matrix(df, percentage, exclude_col, seed = 100):
	'''
	this function will make holes (NAN) in a matrix except the exclude col
	the number of holes is decided by percentage
	'''
	# rows*(col-1), -1 because exclude col
	total_num = df.shape[0]*(df.shape[1]-1)
	holes_num = int(total_num*percentage)
	index_exclude_col =  df.columns.get_loc(exclude_col)
	# randomly generate holes' positions
	r = random.Random(seed)

	col_num = range(1)
	if index_exclude_col == 0:
		col_num = list(range(1, df.shape[1]))
	else:
		col_num = list(range(0, index_exclude_col))+list(range(index_exclude_col+1, df.shape[1]))
	row_num = list(range(0, df.shape[0]))

	count = 0
	select_index_tuple = []
	while count != holes_num:
		row_index = r.choice(row_num)
		col_index = r.choice(col_num)
		if (row_index, col_index) not in select_index_tuple:
			select_index_tuple.append((row_index, col_index))
			df.iloc[row_index, col_index] = np.nan
			count = count + 1


def eval_genomes(genomes, config): #function Used for training model 
# using the training set
    for genome_id, genome in genomes:
        genome.fitness = -1
        net = neat.nn.RecurrentNetwork.create(genome, config)
        for xi, xo in zip(X_train, y_train):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo) ** 2 #Distance from 
            # the correct output summed for all 84 inputs patterns

def r0(filename, threshold, extreme_events_x, extreme_events_y, variable_name='runoff_obs'):
	'''
	this function filter extreme events by using threshold
	'''
	# get data from CSV file
	pd_df = pd.read_csv(filename)
	data = pd_df[variable_name].tolist()
	total_len = len(data)
	# initalize i and j indices
	i = 0
	j = 1
	if abs(data[i]-data[j]) >= threshold:
			extreme_events_x.append(i)
	while j < total_len-1:
		for count in range(total_len-j):
			# this is an extreme event
			if abs(data[i]-data[j]) >= threshold:
				extreme_events_x.append(j)
				extreme_events_y.append(data[j])
				j = j + 1
			# this is a normal event
			else:
				break
		i = j
		j = j + 1


def r1(filename, threshold, extreme_events_x, extreme_events_y, variable_name='runoff_obs'):
	'''
	this function filter extreme events by comparing (tmp_max-tmp_min) and threshold
	'''
	# get data from CSV file
	pd_df = pd.read_csv(filename)
	data = pd_df[variable_name].tolist()
	total_len = len(data)
	# initalize i and j indices
	i = 0
	j = 1
	if abs(data[i]-data[j]) >= threshold:
			extreme_events_x.append(i)
	while j < total_len-1:
		for count in range(total_len-j):
			tmp_max = max(data[i:j+1])
			tmp_min = min(data[i:j+1])
			# this is an extreme event
			if tmp_max - tmp_min >= threshold:
				extreme_events_x.append(j)
				extreme_events_y.append(data[j])
				j = j + 1
			# this is a normal event
			else:
				break
		i = j
		j = j + 1


def r2(filename, threshold, extreme_events_x, extreme_events_y, variable_name='runoff_obs'):
	'''
	this function filter extreme events by using ramp rate threshold
	it is better to normalize our data before use this function
	'''
	# get data from CSV file
	pd_df = pd.read_csv(filename)
	data = pd_df[variable_name].tolist()
	total_len = len(data)
	# initalize i and j indices
	i = 0
	j = 1
	if abs(data[i]-data[j]) >= threshold:
			extreme_events_x.append(i)
	while j < total_len-1:
		for count in range(total_len-j):
			data_diff = abs(data[i]-data[j])
			x_diff = float(j - i)
			# this is an extreme event
			if (data_diff/x_diff) >= threshold:
				extreme_events_x.append(j)
				extreme_events_y.append(data[j])
				j = j + 1
			# this is a normal event
			else:
				break
		i = j
		j = j + 1

def obtain_rc(data, i, j, beta):
	'''
	this function calculates rc value
	'''
	if beta >= 1 or beta < 0:
		print("rc beta should between 0 and 1")
		sys.exit()
	for count in range(j-i+1):
		m = i+count
		pm = data[m]
		# tmp_list is pi, ..., pm
		tmp_list = data[i:m+1]
		if pm <= beta*max(tmp_list):
			return 0
	return 1

def ramp_detect(filename, threshold0, threshold2, extreme_events_x, extreme_events_y, beta, variable_name='runoff_obs'):
	'''
	this method combines r0 and r2; and uses rc
	r1 is not included here because it r1 does not work with our "prms_input_since 2003" data file
	'''
	pd_df = pd.read_csv(filename)
	data = pd_df[variable_name].tolist()
	total_len = len(data)
	# initalize i and j indices
	i = 0
	j = 1
	if abs(data[i]-data[j]) >= threshold0 and abs(data[i]-data[j]) >= threshold2:
			extreme_events_x.append(i)
	while j < total_len-1:
		for count in range(total_len-j):
			# int(True) = 1; int(False) = 0
			r0 = int(abs(data[i]-data[j]) >= threshold0)
			data_diff = abs(data[i]-data[j])
			x_diff = float(j - i)
			r2 = int((data_diff/x_diff) >= threshold2)
			rc = obtain_rc(data, i, j, beta)
			# this is an extreme event
			if rc*r0*r2 == 1:
				extreme_events_x.append(j)
				extreme_events_y.append(data[j])
				j = j + 1
			# this is a normal event
			else:
				break
		i = j
		j = j + 1

def calculate_ramp_score(filename, threshold0, threshold2, beta, variable_name='runoff_obs'):
	'''
	this function calculate ramp score based on W(i,j)
	'''
	return 0

def backward_check_if_extreme_event(filename, threshold0, threshold2, extreme_events_x, extreme_events_y, 
									normal_event_x, normal_event_y, beta, variable_name='runoff_obs'):
	'''
	This function check if the current point is extreme event
	The difference from the "ramp_detect" function is that this
	function check each point backward
	'''
	pd_df = pd.read_csv(filename)
	data = pd_df[variable_name].tolist()
	total_len = len(data)
	# total_len-1 because we don't consider if the first element is an extreme event
	for count in range(total_len-1):
		j = total_len - count - 1
		i = j - 1 
		# TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# should not only consider one point before, should consider a bunch of points before j

		# becuase we do not consider the size of current ramp which ends with j
		# therefore, we only check if the point (i) before j fulfil rc*r0*r1 == 1
		# however, training data is still useful, which is used to tune the parameters,such as beta
		# int(True) = 1; int(False) = 0
		r0 = int(abs(data[i]-data[j]) >= threshold0)
		data_diff = abs(data[i]-data[j])
		x_diff = float(j - i)
		r2 = int((data_diff/x_diff) >= threshold2)
		rc = obtain_rc(data, i, j, beta)
		# this is an extreme event
		if rc*r0*r2 == 1:
			extreme_events_x.append(j)
			extreme_events_y.append(data[j])
		else:
			normal_event_x.append(j)
			normal_event_y.append(data[j])


def vis_two_list(list1_x, list1_y, list2_x, list2_y, fig_title=''):
	'''
	this function vis two lists
	'''
	fig, ax = plt.subplots()
	ax.scatter(list1_x,list1_y, label='normal')
	ax.scatter(list2_x,list2_y, label='extreme')

	legend = ax.legend(loc='upper right', shadow=True)
	# legend = ax.legend(loc='upper right', shadow=True, prop={'size': 20})

	plt.title(fig_title)
	plt.show()

def vis_one_list(list1_x, list1_y, fig_title=''):
	'''
	this function vis one list
	'''
	fig, ax = plt.subplots()
	ax.plot(list1_x,list1_y, label='normal')

	legend = ax.legend(loc='upper right', shadow=True)
	# legend = ax.legend(loc='upper right', shadow=True, prop={'size': 20})

	plt.title(fig_title)
	plt.show()


def vis_all(list1_x, list1_y, list2_x, list2_y, list3_x, list3_y):
	'''
	this function vis two lists
	'''
	fig, ax = plt.subplots()
	ax.scatter(list1_x,list1_y, label='normal')
	ax.scatter(list2_x,list2_y, label='extreme')
	ax.scatter(list3_x,list3_y, label='all')

	legend = ax.legend(loc='upper right', shadow=True, prop={'size': 20})

	plt.show()

def collect_normal(filename, extreme_events_x, normal_event_x, normal_event_y, variable_name='runoff_obs'):
	'''
	'''
	pd_df = pd.read_csv(filename)
	data = pd_df[variable_name].tolist()
	total_len = len(data)
	for count in range(total_len):
		if count not in extreme_events_x:
			normal_event_x.append(count)
			normal_event_y.append(data[count])


def output_list_to_file(filename, list_x):
	'''
	1-D array is outputted into a column of a CSV file
	'''
	with open(filename,'wb') as result_file:
	    for i in list_x:
	    	result_file.write(str(i)+'\n')

def generate_extreme_event_label(extreme_x_label, total_len):
	'''
	this function returns a list (map), if 1 then extreme event
	else normal event
	'''
	result = [0]*total_len
	for count in range(total_len):
		if count in extreme_x_label:
			result[count] = 1
		else:
			result[count] = 0
	return result

def split_file_based_on_threshold(input_file, output_extreme_event_file, output_normal_event_file, 
								   variable_name='runoff_obs', threshold_col_name='threshold_method', label_method='label_method'):
	'''
	This function splits file into two output files based on threshold
	'''
	pd_df = pd.read_csv(input_file)
	# filter
	df_normal = pd_df[pd_df[variable_name]<pd_df[threshold_col_name]]
	df_extreme = pd_df[pd_df[variable_name]>=pd_df[threshold_col_name]]

	# remove threshold and map col
	df_normal_clean = df_normal.drop(columns=[threshold_col_name,label_method])
	df_extreme_clean = df_extreme.drop(columns=[threshold_col_name,label_method])
	# write to file
	df_normal_clean.to_csv(output_normal_event_file,index=False)
	df_extreme_clean.to_csv(output_extreme_event_file,index=False)

def split_file_based_on_label(input_file, output_extreme_event_file, output_normal_event_file, 
								   variable_name='runoff_obs', threshold_col_name='threshold_method', label_method='label_method'):
	'''
	This function splits file into two output files based on label (1 is extreme event)
	'''
	pd_df = pd.read_csv(input_file)
	# filter
	df_normal = pd_df[pd_df[label_method]==0]
	df_extreme = pd_df[pd_df[label_method]==1]

	# remove threshold and map col
	df_normal_clean = df_normal.drop(columns=[threshold_col_name,label_method])
	df_extreme_clean = df_extreme.drop(columns=[threshold_col_name,label_method])
	# write to file
	df_normal_clean.to_csv(output_normal_event_file,index=False)
	df_extreme_clean.to_csv(output_extreme_event_file,index=False)


def accuracy_rate_cal(df, predict_extreme_index, extreme_event_ground_truth_col_name = "Student_Flag", 
	extreme_events_flag = 1, time_col_name = 'Date', extreme_event_col_name = 'NO3N'):
	'''
	extreme_event_ground_truth_col_name: which col is the extreme event flag col
	extreme_events_flag: what value is used to mark extreme events
	This function is designed for only two classes (1 extreme events, 0 normal events) accuracy rate calculation
	'''
	total_len = df.shape[0]
	expert_outlier = df[df[extreme_event_ground_truth_col_name]==extreme_events_flag]
	ground_truth_index = expert_outlier.index.to_numpy()
	ground_truth = []
	ground_truth_extreme_event_timestamp = []
	ground_truth_extreme_event_values = []
	predict_label = []
	predict_extreme_event_timestamp = []
	predict_extreme_event_vales = []

	# if we have more than 2 classes, need to update this for loop
	for i in range(total_len):
		if i in ground_truth_index:
			# if ground_truth_index has the counter
			# it means it is an anomaly event
			ground_truth.append(1)
			ground_truth_extreme_event_timestamp.append(df.iloc[i][time_col_name])
			ground_truth_extreme_event_values.append(df.iloc[i][extreme_event_col_name])
		else:
			ground_truth.append(0)
		if i in predict_extreme_index:
			predict_label.append(1)
			predict_extreme_event_timestamp.append(df.iloc[i][time_col_name])
			predict_extreme_event_vales.append(df.iloc[i][extreme_event_col_name])
		else:
			predict_label.append(0)

	print("sum ground_truth: ", sum(ground_truth))
	print("sum predict_label: ", sum(predict_label))
	# accuracy_score is calculated with (TP+TN)/(TP+TN+FP+FN)
	print("Accuracy Score is: ", accuracy_score(ground_truth, predict_label))
	return ground_truth_extreme_event_timestamp,ground_truth_extreme_event_values,predict_extreme_event_timestamp,predict_extreme_event_vales

def vis_normal_events_and_extreme_events(df, ground_truth_extreme_event_timestamp, 
	ground_truth_extreme_event_values, predict_extreme_event_timestamp, 
	predict_extreme_event_vales, time_col_name = 'Date', extreme_event_col_name = 'NO3N'):
	'''
	this function visualize normal events and extreme events
	df should have time col
	'''
	time = df[time_col_name]
	# parse function can automatically convert time str
	# to datetime
	# 8/2/2016 12:29
	x_time = [parse(x) for x in time]
	values = df[extreme_event_col_name]
	# rotate the x axis title or overlap
	plt.xticks(rotation=45)
	# preview first 500 or too big 
	plt.scatter(x_time, values, label=extreme_event_col_name)
	plt.scatter(ground_truth_extreme_event_timestamp, ground_truth_extreme_event_values, facecolors='none', edgecolors='r', label="Ground Truth Extreme Event")
	plt.scatter(predict_extreme_event_timestamp, predict_extreme_event_vales, marker='^', label="Predicted Extreme Events")
	plt.legend()
	plt.show()

def count_nan_for_each_col(df):
	'''
	'''
	print("number of nan:")
	print(df.isna().sum())
	print("Percentage of nan:")
	print(df.isna().sum()/len(df.index))
	print("total len is: ", len(df.index))

def get_time_stamp_array(df, time_col='timestamp'):
	'''
	this function is used to convert
	string to datetime
	'''
	return pd.to_datetime(df[time_col])

def gen_holes(df, df2, total_num=100):
	'''
	generate a random number [x,y]
	0<x<n1; 0<y<n2; df.shape = (n1,n2)
	no repeated pair
	make holes (nan) in df based on ramdom number pairs
	return df, a dictionary about holes and original number
	The reason we input two df copies because df is likt a pointer
	when we df.iloc[x,y]=np.nan, it will impact 'value':df2.iloc[x,y]
	'''
	# -1 because index starting from 0 for df elements
	n1 = df.shape[0] - 1
	n2 = df.shape[1] - 1
	result_list = []

	count = 0
	while count < total_num:
		x, y = randint(0, n1), randint(0, n2)
		if [x,y] not in result_list:
			# if df.iloc[x,y] == np.nan:
			# 	print(x,y)
			# 	return
			# print("values: {} for {}, {}, type is {}".format(df.iloc[x,y],x,y, type(df.iloc[x,y])))
			result_list.append({'coordinate': [x,y], 'value':df2.iloc[x,y]})
			count = count + 1
			# make holes
			df.iloc[x,y]=np.nan
	return df, result_list

def accuracy_checking(result_np_array, ground_truth_list):
	'''
	df should be filled with predicted values for all nan holes
	ground_truth_list should be a list of dicts {'coordinate': [x,y], 'value':df.iloc[x,y]}
	'''
	prediction = []
	ground_truth = []
	for element in ground_truth_list:
		curr_coordinate_x = element['coordinate'][0]
		curr_coordinate_y = element['coordinate'][1]
		prediction.append(result_np_array[curr_coordinate_x, curr_coordinate_y])
		ground_truth.append(element['value'])

	# test vis
	# x_index = []
	# for i in range(len(ground_truth)):
	# 	x_index.append(i)

	# plt.scatter( x_index, prediction, label="prediction")
	# plt.scatter( x_index, ground_truth, label="ground_truth")
	# plt.xlabel('Index')
	# plt.ylabel('Values')
	# plt.show()

	print("R^2 is: ", r2_score(ground_truth, prediction))
	return prediction, ground_truth

def get_nse(list1,list2):
	'''
	Nash-Sutcliffe efficiency
	list1 is model simulated value
	list2 is observed data
	'''
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff_power = 0
	sum_diff_o_power = 0
	mean_list2 = sum(list2) / len(list2)
	for count in range(list_len):
		sum_diff_power = sum_diff_power + (list1[count]-list2[count])**2
		sum_diff_o_power = sum_diff_o_power + (list2[count]-mean_list2)**2
	result = sum_diff_power/sum_diff_o_power
	return 1 - result

def get_avg_gap_size(df, col_name):
	'''
	this function calculate avg missing data gap length
	for col with the input name (col_name)
	'''
	# remove all holes
	df_no_nan = df[col_name].dropna()
	index_without_holes = df_no_nan.index.tolist()
	# calculate difference between adjacent elements
	# without gap, it should be 1, with gap it will be gap size + 1
	diff_list = []
	for i in range(1, len(index_without_holes)):
		# -1 because without gap, it should be 1, with gap it will be gap size + 1
		diff_list.append(index_without_holes[i] - index_without_holes[i-1] - 1)

	# this part is for the last gap (from n to the end of df)
	total_len = df.shape[0]
	# total_len - 1 is the last element index
	diff_list.append(total_len - 1 - index_without_holes[-1])
	# remove all 0s from the list
	gap_size = [i for i in diff_list if i != 0]
	if gap_size == []:
		# no gap
		return 0

	avg_gap_size = sum(gap_size)/len(gap_size)
	print("Avg gap for {} is {}".format(col_name, avg_gap_size))
	return avg_gap_size

# CCRM starts from here
# this function is originally from https://code.sololearn.com/cqMD5wu2rhUJ/#py
def factors(n):
    a =[1]
    for i in range(2,(n+1),1):
        if(n%i==0):
            a = a+[i]
    return a

def check_if_df_col_all_0(df):
	'''
	this function checks if a df contains a col with only zeros
	inspired by the code at: https://stackoverflow.com/questions/21164910/how-do-i-delete-a-column-that-contains-only-zeros-in-pandas
	'''
	# return results for each col, if all 0s then False, if not all 0s true
	check_col_results = (df != 0).any(axis=0)
	return False in check_col_results.to_list()

def reshape(df, curr_factor):
	'''
	reshape df based on current factor
	'''
	# first cur_factor rows, first col
	result_df = df.iloc[:curr_factor,0].copy()
	n1 = df.shape[0]
	n2 = df.shape[1]
	quotient = int(n1/curr_factor)
	# used to check if this is the first time concat
	flag_first_time = True
	for curr_col in range(n2):
		# range(1,total_merge_time) => 1, 2, 3, ..., total_merge_time - 1
		for curr_row in range(quotient):
			if flag_first_time:
				# skip the first time combination, because it is already inital result_df
				flag_first_time = False
			else:
				curr_df = df.iloc[curr_row*curr_factor:(curr_row+1)*curr_factor,curr_col].copy().reset_index(drop=True)
				result_df = pd.concat([result_df, curr_df], ignore_index=True, axis=1)

	return result_df

def ccrm(df, input_factor=-1):
	'''
	based on the paper An alternating direction method of 
	multipliers based approach for pmu data recovery
	this function is used to reshape a matrix (n1 by n2)
	if factor==-1, then use the first usable factor,
	if not, use the input one
	'''
	n1 = df.shape[0]
	n2 = df.shape[1]
	n1_factors = factors(n1)
	n1_factor_decent = sorted(n1_factors, reverse=True)
	# do not check 1 and itself
	check_factors = n1_factor_decent[1:-1]

	qualified_factor = []
	# check which factors work
	for factor in check_factors:
		# this rule is a little bit different from paper algorithm 2
		# I added =
		# if (n1/factor>=n2) and math.ceil(n1/(factor+1))<=n2:
		# 	qualified_factor.append(factor)
		# removed the rules for now...///////////////!!!!
		qualified_factor.append(factor)

	print("Qualified factors include: "+str(qualified_factor))
	# if no qualified factor
	if qualified_factor == []:
		print("no qualified factor in n1")
		return -1

	if input_factor == -1:
		# reshape, I only return the first possible df
		for i in qualified_factor:
			curr_reshape = reshape(df, i)
			if check_if_df_col_all_0(curr_reshape) == False:
				# this means curr_reshape is one of the possible qualified reshape
				print("After reshape, the current shape is:"+str(curr_reshape.shape))
				return curr_reshape, i
			else:
				print("Warning: Factor "+str(i)+
					" causes at least one col with all 0s.")
				return curr_reshape, i
	elif input_factor in qualified_factor:
		# reshape with the input_factor
		curr_reshape = reshape(df, input_factor)
		if check_if_df_col_all_0(curr_reshape) == False:
			# this means curr_reshape is one of the possible qualified reshape
			print("After reshape, the current shape is:"+str(curr_reshape.shape))
			return curr_reshape, input_factor
		else:
			print("Warning: Factor "+str(input_factor)+
					" causes at least one col with all 0s.")
			return curr_reshape, input_factor
	else:
		print("Error: input_factor is not one of the qualified ones.")
		return -1
	print("no possible reshape based on the ccrm rules")
	return -1

def ccrm_reverse(reshape_df, original_n1, original_n2):
	'''
	this function reverses the ccrm reshape results to the original
	matrix, original_n1 is the original rows count
	original_n2 is the original cols count
	factor is the factor used for reshape in ccrm and it should be
	reshape_df.shape[0]
	'''
	n1 = reshape_df.shape[0]
	n2 = reshape_df.shape[1]
	each_col_loop_times = int(original_n1/n1)

	for curr_col_id in range(original_n2):
		# prepare the original col
		starting_point = curr_col_id*each_col_loop_times
		result_col = reshape_df.iloc[:,starting_point]
		for curr_col_chunk_id in range(1,each_col_loop_times):
			curr_chunk = reshape_df.iloc[:,starting_point+curr_col_chunk_id]
			result_col = pd.concat([result_col,curr_chunk], ignore_index=True, axis=0)
		# first col
		if curr_col_id == 0:
			result_df = result_col
		else:
			result_df = pd.concat([result_df, result_col], ignore_index=True, axis=1)

	return result_df

# CCRM ends here

# row based reshape
def row_based_reshape(df, input_factor=-1):
	'''
	this function is different from CCRM
	CCRM reshape a matrix col by col
	this function reshape a matrix row by row
	this row based reshape can be thinked as reverse of ccrm with transpose(df)
	try to draw a 6*4 matrix and reshape it to 2*12, it will be clearer
	'''
	n1 = df.shape[0]
	n2 = df.shape[1]
	n1_factors = factors(n1)
	n1_factor_decent = sorted(n1_factors, reverse=True)
	# do not check 1 and itself
	check_factors = n1_factor_decent[1:-1]

	qualified_factor = []
	# check which factors work
	for factor in check_factors:
		# this rule is a little bit different from paper algorithm 2
		# I added =
		# if (n1/factor>=n2) and math.ceil(n1/(factor+1))<=n2:
		# 	qualified_factor.append(factor)
		# removed the rules for now...///////////////!!!!
		qualified_factor.append(factor)

	print("Qualified factors (lags) include: "+str(qualified_factor))
	# if no qualified factor
	if qualified_factor == []:
		print("no qualified factor in n1")
		return -1

	df_transpose = df.transpose()
	# reorder every element from minimum to maximum
	qualified_factor.sort()
	if input_factor == -1:
		# reshape, I only return the first possible df
		for i in qualified_factor:
			# i is the factor
			curr_reshape = ccrm_reverse(df_transpose, int(n2*i) , int(n1/i))
			if check_if_df_col_all_0(curr_reshape) == False:
				# this means curr_reshape is one of the possible qualified reshape
				print("After reshape, the current shape is:"+str(curr_reshape.transpose().shape))
				return curr_reshape.transpose(), i
			else:
				print("Warning: Factor "+str(i)+
					" causes at least one col with all 0s.")
				return curr_reshape.transpose(), i
	elif input_factor in qualified_factor:
		# reshape with the input_factor
		curr_reshape = ccrm_reverse(df_transpose, int(n2*input_factor) , int(n1/input_factor))
		if check_if_df_col_all_0(curr_reshape) == False:
			# this means curr_reshape is one of the possible qualified reshape
			print("Orignal matrix shape is:"+str(n1)+", "+str(n2))
			print("After reshape, the current shape is:"+str(curr_reshape.transpose().shape))
			return curr_reshape.transpose(), input_factor
		else:
			print("Warning: Factor "+str(input_factor)+
					" causes at least one col with all 0s.")
			return curr_reshape.transpose(), input_factor
	else:
		print("Error: input_factor is not one of the qualified ones.")
		return -1
	print("no possible reshape based on the ccrm rules")
	return -1

def row_based_reshape_reverse(reshape_df, original_n1, original_n2):
	'''
	this function reverses the row based reshape results to the original
	matrix, original_n1 is the original rows count
	original_n2 is the original cols count
	'''
	factor = original_n2
	reshape_df_tran = reshape_df.transpose()
	try:
		curr_reshape = reshape(reshape_df_tran, factor)
	except:
		print("Reshape does not work.")
		raise
	return curr_reshape.transpose()

# row based reshape ends here

def draw_autocorr(input_np_arr, input_maxlags=1000):
	'''
	this function visualizes the autocorrelations
	of an input array
	e.g. input can be df['q_cms'].values
	'''
	plt.acorr(input_np_arr, maxlags=input_maxlags)
	plt.title('Autocorrelation VS lag')
	plt.xlabel('Lag (number of rows)')
	plt.ylabel('Autocorrelation')
	# Display the autocorrelation plot
	plt.show()

def randomly_create_continous_gap(df, gap_size, col_name, extreme_events_arr):
	'''
	this function will randomly create a continous gap 
	for one variable of a df with the size of gap_size_percent*len
	'''
	total_len = len(extreme_events_arr)
	# from front to back
	start_index = random.randint(0,total_len-gap_size-1)
	end_index = start_index + gap_size
	
	for index in extreme_events_arr[start_index:end_index]:
		df[col_name].iloc[index] = np.nan

	return df, start_index, end_index
