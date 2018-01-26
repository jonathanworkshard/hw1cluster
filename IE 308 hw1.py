#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 17:52:17 2018

@author: jonathan shenkman

IE 308 hw 1

clustering medicare healthcare providers


features not relevant:
    -name of providers
    -street address
    -zip code
        -too many categorical variables
    
relevant:
    -credentials
    -location
    -# of records for them?
    -gender
    
objective function: some cost metric relative to number of patients served/service offered
    -clustering will take care of 
    
data to add:
    -features related to location that could impact health results


areas to immprove
    -right now, just taking the first degree in credentials
    -in future, run onehotencoder on each potential degree position and take the union
    -scale relative importance of features
    -more sophisticated outlier detection
    -get outside data to quantify/act as a proxy for categorical data that couldnt be included (zip code and )
    


ideas
-include charged and max allowed amount - see who they paid too much
-include everything but charged, see what doctors are charging too much
-include no money info, see overactive doctors

questions
-is standard deviation of target in clustering a good measure of effectiveness?
-does standardization/normalization change the silhouette?
    -do you need to do same type of feature transformation for all variables?
-how do you know what level to use pca?
    
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn import cluster
from collections import Counter
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import sys

# set up logging
old_stdout = sys.stdout

log_file = open("hw1.log","w")

sys.stdout = log_file




# step 1: understanding the data and finding correlations

def get_data():
    """
    function to retrieve data from file
    """
    data = pd.read_table('Medicare_Provider_Util_Payment_PUF_CY2015.txt')
    # clean out first row
    data = data.iloc[1:]
    data = data.fillna('nan')
    return data


def string_hist(data):
    """
    makes histogram out of column of strings
    Input:
        data: column of strings
    
    """
    letter_counts = Counter(data)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')
    df.plot(kind='bar')
    print(letter_counts)
    return letter_counts

    
# separate numeric and categorical variables
  
    
# group the hcpcs codes


numeric_use = ['line_srvc_cnt',
       'bene_unique_cnt', 'bene_day_srvc_cnt','average_submitted_chrg_amt']

categorical_use = ['nppes_credentials','nppes_provider_gender','nppes_entity_code',
                   'nppes_provider_city','nppes_provider_zip', 'nppes_provider_state',
                   'nppes_provider_country','provider_type','medicare_participation_indicator','place_of_service',
                   'hcpcs_code','hcpcs_drug_indicator']

categorical_use_base = ['nppes_credentials','nppes_provider_gender','nppes_entity_code',
                   'nppes_provider_state',
                   'nppes_provider_country','provider_type','medicare_participation_indicator','place_of_service',
                   'hcpcs_code','hcpcs_drug_indicator']

categorical_use_base_updated = ['nppes_credentials','nppes_provider_gender','nppes_entity_code',
                   'nppes_provider_state','provider_type','place_of_service','hcpcs_code',
                   'hcpcs_drug_indicator']
      
outputs = ['average_Medicare_allowed_amt', 'average_Medicare_payment_amt',
       'average_Medicare_standard_amt']
outputs = ['average_Medicare_payment_amt']


def make_categorical(data):
    """
    takes a column of string data and makes it into dummy variables
    """
    # replace Nan with a string
    data = data.fillna('nan')
    # fit strings with integer values
    label_encoder = preprocessing.LabelEncoder()
    fitted = label_encoder.fit_transform(data)
    # store the diff classes for column names
    classes = label_encoder.classes_
    # convert integers to binary dummy variables
    enc = preprocessing.OneHotEncoder()
    # reshape the fitted properly
    fitted = fitted.reshape(-1,1)
    enc_fit = enc.fit_transform(fitted).toarray()
    return enc_fit,classes

def simplify_degree(degree,thresh = 50000):
    """
    
    """
    # filter out nan
    degree = degree.fillna('nan')
    # take only the first degree
    first_degree = [element.split(',')[0] for element in degree]
    string_simplified = [string.replace('.','').lower() for string in first_degree]
    # only take the more common degrees
    #keep_credentials = detect_categorical_outliers(string_simplified,thresh)
    
    return string_simplified


def detect_categorical_outliers(string_list,thresh):
    """
    creates a boolean True for values that occur at frequency above thresh.  False otherwise
    Input:
        string_list: a list of strings for analysis
        thresh: number of repeats to be considered non-outlier
    Output:
        keep_credentials: boolean.  True if value occurs above thresh times, false otherwise
    """
    # only take the more common degrees
    count = Counter(string_list)
    above_thresh = [element > thresh for element in list(count.values())]
    # get the names of credentials above thresh
    above_thresh_credentials = [list(count.keys())[element] for element in np.where(above_thresh)[0]]
    # if abovee thresh, keep value.  Else, make note to delete entry
    keep_credentials = [element in above_thresh_credentials for element in string_list]
    return keep_credentials


def diff_values(data):
    """
    get the number of different categorical variables/unique values for columns
    Input: 
        data: column of data
    """
    count = Counter(data)
    num_points = len(count)
    return num_points
    
def eliminate_categorical_outliers(data,categorical_variables,thresh=50000):
    """
    takes the data dataframe and gets rid of rows that are considered outliers
    Input:
        data: pandas dataframe with all data
        categorical_variables: list of the names of the variables that will be
        used as the categorical variables.  Only looks at outliers from these
        thresh: the threshold below which a category is considered an outlier
    
    """
    # initialize boolean
    keep_total = np.repeat(True,data.shape[0])
    
    # iterate over categories and edit to False if any new column says outlier/False
    for category in categorical_variables:
        keep_credentials = detect_categorical_outliers(data[category],thresh)
        keep_total = keep_total & keep_credentials
    data = data.iloc[np.where(keep_total)]
    return data

def clean_numeric(data,numeric_variables,thresh =2):
    """
    """
    scaled = preprocessing.scale(data[numeric_variables])
    # outlier if more than two standard deviations away.
    keep_numeric = np.abs(scaled) < 2
    # make sure true for all columns
    
    keep_numeric = keep_numeric[:,0]&keep_numeric[:,1]&keep_numeric[:,2]
    # put scaled into dataframe to replace old values
    data[numeric_variables] = scaled
    # get rid of entries that have outliers
    data = data.iloc[np.where(keep_numeric)]
    return data

def prep_factors_final(data,categorical_variables):
    """
    makes does onehotencoder for each of the categorical variables and keeps
    track of the new dummy variables.  Merges all of the 
    Input:
        data: the pd series with all the outliers already removed
        caegorical_variables: list of the strings that are names of the 
        categorical variable columns in data
    
    Output: 
        dummy_var_frame: pandas data frame with the categorical variables
        converted into dummy variables
    """
    # start with first variable to initialize
    dummy_variables,variable_names = make_categorical(data[categorical_variables[0]])
    # add the categorical variable name to the individual names
    variable_names = categorical_variables[0] + '_' + variable_names
    # loop through rest of categorical variables
    for category in categorical_variables[1:]:
       temp_dummy_variables,temp_variable_names = make_categorical(data[category])
       # add the categorical variable name to the individual names
       temp_variable_names = category + '_' + temp_variable_names
       # merge with the overall variables
       dummy_variables = np.concatenate([dummy_variables,temp_dummy_variables],axis=1)
       variable_names = np.concatenate([variable_names,temp_variable_names])
    dummy_var_frame = pd.DataFrame(data=dummy_variables,columns = variable_names)
    return dummy_var_frame

def random_sample(data,size):
    """
    take a subset of the data to run kmeans
    Input:
        data: the prepped dataframe for kmeans
        size: number of observations for the sample
    """
    
    random_index = np.random.randint(data.shape[0],size = size)
    data_subset = data.loc[random_index]
    return data_subset


def take_correlated(data,cutoff =.05):
    """
    takes correlation with output (last column) and only keeps variables with correlation
    above cutoff value
    
    Input:
        data: input dataframe.  All numeric.  Last column is output
        cutoff: the absolute value of correlation needed to be considered as a feature
    Output:
        data: Only the features with correlation above the cutoff.  Output not included
    """
    feature_correlation = data.corr()
    # get rid of last row (autocorrelation)
    feature_correlation = feature_correlation.iloc[:-1,:]
    #plot the magnitude of the correlation
    sorted_data = np.sort(feature_correlation['average_Medicare_payment_amt'].abs().values)
    
    yvals=np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data,yvals)
    plt.show()
    # get name of variables with high correlation
    high_cor_index = np.where(feature_correlation['average_Medicare_payment_amt'].abs().values> cutoff)[0]
    high_cor_features = [list(feature_correlation.index)[element] for element in high_cor_index]
    # take the features with high correlation
    data = data[high_cor_features]
    return data

def iterate_kmeans_score(var_frame,n_clusters,num_iter = 10,size_break = 100000):
    """
    """
    # initialize results
    results = 0
    for i in np.arange(num_iter):
        data_subset = random_sample(var_frame,size_break)
        # convert to numpy array
        data_numpy = np.array(data_subset)
        
        #initialize KMeans
        kmeans = cluster.KMeans(n_clusters=n_clusters)
        kmeans.fit(data_numpy)
        results += -kmeans.score(var_frame)
    return results

def best_n_cluster(var_frame,max_clusters):
    """
    creates cluster and evaluates for [2,max_clusters] k values
    
    Inputa
    """
    x_val = list()
    y_val = list()
    silhouette = list()
    for i in np.arange(2,max_clusters+1):
        print('on cluster number', i)
        # sum_score = iterate_kmeans_score(var_frame,i)
        minibatch = cluster.MiniBatchKMeans(n_clusters=i,batch_size = 50000,n_init=10)
        minibatch.fit(var_frame)
        sum_score = minibatch.score(var_frame)
        x_val.append(i)
        y_val.append(sum_score)
        # get labels
        labels = minibatch.predict(var_frame)
        silhouette.append(get_silhouette(var_frame,labels))
    return x_val,y_val, silhouette


def get_silhouette(var_frame,labels):
    """
    gets the average silhouette score for the inputs
    Inputs:
        var_frame: pd dataframe of features using
        labels: labels of the objects in var_frame
    output: 
        mean_score: the mean silhouette score of the iterations
    """
    score = list()
    for i in np.arange(5):
        # get silhouette score
        score.append(sklearn.metrics.silhouette_score(var_frame,labels,sample_size=10000))
    # take the mean
    mean_score = np.mean(score)
    return mean_score



def group_hcpcs(hcpcs_codes):
    """
    """    
    # if ends in 'T', then category3
    """
    for i in np.arange(len(hcpcs_codes)):
        if hcpcs_codes.iloc[i][-1] == 'T':
            hcpcs_codes.iloc[i] = 'cat3'
        elif hcpcs_codes.iloc[i][0] == '1':
            hcpcs_codes.iloc[i] = 'surgery integumentary'
        elif hcpcs_codes.iloc[i][0] == '2':
            hcpcs_codes.iloc[i] = 'surgery musculoskeletal'
        elif hcpcs_codes.iloc[i][0] == '3':
            hcpcs_codes.iloc[i] = 'surgery3'
        elif hcpcs_codes.iloc[i][0] == '4':
            hcpcs_codes.iloc[i] = 'surgery digestive'
        elif hcpcs_codes.iloc[i][0] == '5':
            hcpcs_codes.iloc[i] = 'Surgery genitalia'
        elif hcpcs_codes.iloc[i][0] == '6':
            hcpcs_codes.iloc[i] = 'surgery nervous'
        elif hcpcs_codes.iloc[i][0] == '0':
            hcpcs_codes.iloc[i] = 'anesthesia'
        elif hcpcs_codes.iloc[i][0] == '7':
            hcpcs_codes.iloc[i] = 'radiology'
        elif hcpcs_codes.iloc[i][0] == '8':
            hcpcs_codes.iloc[i] = 'pathology'
        elif hcpcs_codes.iloc[i][0] == '9':
            hcpcs_codes.iloc[i] = 'services'
        elif hcpcs_codes.iloc[i][0].isalpha():
            hcpcs_codes.iloc[i] = 'supplemental'
        else:
            print('what is this')
    """
    is_el = [element[0].isalpha() for element in data['hcpcs_code']]
    hcpcs_codes[is_el] = 'supplemental'
    is_el = [element[-1] == 'T' for element in data['hcpcs_code']]
    hcpcs_codes[is_el] = 'cat3'
    is_el = [element[0] == '0' for element in data['hcpcs_code']]
    hcpcs_codes[is_el] = 'anesthesia'
    is_el = [element[0] == '1' for element in data['hcpcs_code']]
    hcpcs_codes[is_el] = 'surgery integumentary'
    is_el = [element[0] == '2' for element in data['hcpcs_code']]
    hcpcs_codes[is_el] = 'surgery musculoskeletal'
    is_el = [element[0] == '3' for element in data['hcpcs_code']]
    hcpcs_codes[is_el] = 'surgery3'
    is_el = [element[0] == '4' for element in data['hcpcs_code']]
    hcpcs_codes[is_el] = 'surgery digestive'
    is_el = [element[0] == '5' for element in data['hcpcs_code']]
    hcpcs_codes[is_el] = 'Surgery genitalia'
    is_el = [element[0] == '6' for element in data['hcpcs_code']]
    hcpcs_codes[is_el] = 'surgery nervous'
    is_el = [element[0] == '7' for element in data['hcpcs_code']]
    hcpcs_codes[is_el] = 'radiology'
    is_el = [element[0] == '8' for element in data['hcpcs_code']]
    hcpcs_codes[is_el] = 'pathology'
    is_el = [element[0] == '9' for element in data['hcpcs_code']]
    hcpcs_codes[is_el] = 'services'

    
    return hcpcs_codes
            
def get_ids(data,frame_used,labels):
    """
    retrieves the npis for points that have been assigned to certain clusters
    """
    # initialize dict
    cluster_ids = {}
    for element in np.unique(labels):
        # get index within frame_used
        id_index = np.where(labels == element)[0]
        # get actual index for origin
        id_numbers_index = np.array(frame_used.index)[id_index]
        cluster_ids[element] = data.loc[id_numbers_index]['npi']
    return cluster_ids

def get_payments(data,frame_used,labels):
    """
    retrieves the standardized payment for points that have been assigned to certain clusters
    """
    # initialize dict
    cluster_payments = {}
    for element in np.unique(labels):
        # get index within frame_used
        id_index = np.where(labels == element)[0]
        # get actual index for origin
        id_numbers_index = np.array(frame_used.index)[id_index]
        cluster_payments[element] = data.loc[id_numbers_index]['average_Medicare_payment_amt']
    return cluster_payments

def count_repetitions(cluster_ids):
    """
    """
    size_cluster = [len(element) for element in cluster_ids.values()]
    n_unique_cluster = [len(np.unique(element)) for element in cluster_ids.values()]
    prob_unique = np.array(n_unique_cluster)/np.array(size_cluster)
    #print('average number of provider repetitions per cluster')
    #plt.hist(ave_repetitions)
    #plt.plot(ave_repetitions)
    return prob_unique

def get_pseudo_partial_f(cluster_payments):
    """
    returns variance within cluster divided by variance outside of cluster
    """
    cluster_variance = [np.var(element) for element in list(cluster_payments.values())]
    #get overall variance
    all_payments = np.concatenate(list(cluster_payments.values()))
    overall_variance = np.var(all_payments)
    # get pseudo f
    pseudo_f = overall_variance/np.array(cluster_variance)
    #plt.hist(pseudo_f_payments,bins=[-1,0,1,5,10,20,30,40,50,100])

    return pseudo_f

def useful_cluster_plot(cluster_ids, cluster_payments):
    """
    
    """
    prob_unique=count_repetitions(cluster_ids)
    pseudo_f = get_pseudo_partial_f(cluster_payments)
    
    plt.scatter(pseudo_f,prob_unique)
    plt.axhline(prob_unique_overall, color="red", linestyle="--")
    plt.show()
    return pseudo_f,prob_unique

data = get_data()
# clean and replace the degree
data['nppes_credentials'] = simplify_degree(data['nppes_credentials'])
# number of different HSPC codes
num_codes = len(data['hcpcs_code'].unique())
print('there are %d different HCPCS codes shown'  %num_codes)
# group codes by category
data['hcpcs_code'] = group_hcpcs(data['hcpcs_code'])

#initialize dataframe to store occurences
counts = pd.DataFrame(index=categorical_use,columns = ['count'])


# get the number of points for each of the categorical vars
for categorical in categorical_use:
    counts.loc[categorical] = diff_values(data[categorical])
print(counts)
# for base case, use the ones with fewer categories
# get rid of outliers
new_data = eliminate_categorical_outliers(data,categorical_use_base)
# get new number of categories
print('new number of categories after outlier removal')
counts = pd.DataFrame(index=categorical_use_base,columns = ['count'])
for categorical in categorical_use_base:
    counts.loc[categorical] = diff_values(new_data[categorical])
print(counts)

# nppes_provider_country and medicare_participation_indicator only have 1 category now
# show histograms for these
string_hist(data['nppes_provider_country'])
string_hist(data['medicare_participation_indicator'])
# almost no data for other countries, but still 3440 examples of non-providers
# might have a large impact so should be aware and maybe test this

# get rid of points with numeric outliers and scale data
new_data = clean_numeric(new_data,numeric_use)

# initialize numpy array and 
# make categorical variables into binary
dummy_var_frame = prep_factors_final(new_data,categorical_use_base_updated)
# combine numeric with categorical variables
numeric_frame_use = new_data[numeric_use]
dummy_var_frame.index = numeric_frame_use.index
var_frame = pd.concat([numeric_frame_use,dummy_var_frame],axis=1)


# take out nppes_provider_gender_nan  and nppes_entity_code_I bc data already included in nppes_entity_code_O
var_frame = var_frame.drop(['nppes_provider_gender_nan', 'nppes_entity_code_I'],axis=1)


# create another dataframe with the outputs to check for correlation
output_frame_use = pd.DataFrame(preprocessing.scale(new_data[outputs]),index=var_frame.index,columns = outputs)


#combine the frames
test_frame = pd.concat([var_frame,output_frame_use],axis=1)

####### different methods for choosing features
# type 1: correlation
# around 20% have abs correlation over .05.  use these
cor_var_frame = take_correlated(test_frame)


# type 2: pca
pca = decomposition.IncrementalPCA()
pca_frame_dict=pca.fit_transform(var_frame)

# make frame for pca cutoff value of .5% of explained variance total
num_components = np.sum(np.cumsum(pca.explained_variance_ratio_) < .5)

pca = decomposition.IncrementalPCA(n_components=num_components)
pca_frame_dict = pca.fit_transform(var_frame)
pca_frame_dict = pd.DataFrame(pca_frame_dict,index=var_frame.index)

####### get results
x_val,y_val, silhouette = best_n_cluster(var_frame,10)
x_val_cor1,y_val_cor1, silhouette_cor1 = best_n_cluster(cor_var_frame,60)
x_val_pca1,y_val_pca1, silhouette_pca1 = best_n_cluster(pca_frame_dict,60)
#x_val_pca2,y_val_pca2, silhouette_pca2 = best_n_cluster(pca_frame_dict[47],22)
#x_val_pca3,y_val_pca3, silhouette_pca3 = best_n_cluster(pca_frame_dict[7],22)
#x_val_pca4,y_val_pca4, silhouette_pca4 = best_n_cluster(pca_frame_dict[.9],22)


# take subset of data, evaluate, and average results
# compare results

plt.plot(x_val_cor1[:], y_val_cor1[:], label='corelation loss')
plt.plot(x_val[:], y_val[:], label='base loss')
plt.plot(x_val_pca1[:], y_val_pca1[:], label='pca1 loss')
#plt.plot(x_val_pca2[:8],y_val_pca2[:8],label='pca2 loss')
#plt.plot(x_val_pca3[:8],y_val_pca3[:8],label='pca3 loss')

plt.legend()
plt.show()


plt.plot(x_val_cor1,silhouette_cor1,label= 'corelation silhouette')

plt.plot(x_val,silhouette,label='base silhouette')
plt.plot(x_val_pca1,silhouette_pca1,label='pca1 silhouette')
#plt.plot(x_val_pca3,silhouette_pca3,label='pca3 silhouette')
#plt.plot(x_val_pca4,y_val_pca4,label='pca4 loss')
#plt.plot(x_val_pca4,silhouette_pca4,label='pca4 silhouette')
plt.legend()
plt.show()











# return the income info for each cluster
# calculate overall probability of unique
prob_unique_overall = new_data['npi'].nunique()/new_data.shape[0]


# cluster each of the cases with 5 clusters and compare payment results
minibatch = cluster.MiniBatchKMeans(n_clusters=5,batch_size = 100000,n_init=10)
labels_base= minibatch.fit_predict(var_frame)
cluster_id_base = get_ids(data,var_frame,labels_base)
cluster_payments_base = get_payments(data,var_frame,labels_base)
pseudo_f_base,prob_unique_base = useful_cluster_plot(cluster_id_base,cluster_payments_base)


minibatch = cluster.MiniBatchKMeans(n_clusters=5,batch_size = 100000,n_init=10)
labels_cor= minibatch.fit_predict(cor_var_frame)
cluster_id_cor = get_ids(data,cor_var_frame,labels_cor)
cluster_payments_cor = get_payments(data,cor_var_frame,labels_cor)
pseudo_f_cor,prob_unique_cor = useful_cluster_plot(cluster_id_cor,cluster_payments_cor)


minibatch = cluster.MiniBatchKMeans(n_clusters=5,batch_size = 100000,n_init=10)
labels_pca= minibatch.fit_predict(pca_frame_dict)
cluster_id_pca = get_ids(data,pca_frame_dict,labels_pca)
cluster_payments_pca = get_payments(data,pca_frame_dict,labels_pca)
pseudo_f_pca,prob_unique_pca = useful_cluster_plot(cluster_id_pca,cluster_payments_pca)


# cluster correlated with 30 batches and compare
minibatch = cluster.MiniBatchKMeans(n_clusters=30,batch_size = 100000,n_init=10)
labels_cor= minibatch.fit_predict(cor_var_frame)
cluster_id_cor = get_ids(data,cor_var_frame,labels_cor)
cluster_payments_cor = get_payments(data,cor_var_frame,labels_cor)
pseudo_f_cor,prob_unique_cor = useful_cluster_plot(cluster_id_cor,cluster_payments_cor)

# take out outliers and regraph to interpret
outlier = np.argmax(pseudo_f_cor)
pseudo_f_cor_out = np.delete(pseudo_f_cor,outlier)
prob_unique_cor_out = np.delete(prob_unique_cor,outlier)
plt.scatter(pseudo_f_cor_out,prob_unique_cor_out)
plt.axhline(prob_unique_overall, color="red", linestyle="--")
plt.show()

# normalize the clusters above the overall prob unique and above 20x f and look for outliers
above_ave = prob_unique_cor>prob_unique_overall
above20 = pseudo_f_cor > 20
great_cluster=  above_ave & above20
great_cluster_keys = [list(cluster_payments_cor.keys())[element] for element in np.where(great_cluster)[0]]
# go through all of the good cluster and analyze the payment info
# intiialize number 3td away
std3 = 0
std3_index = list()
std3_savings = 0
for key in great_cluster_keys:
    payment = cluster_payments_cor[key]
    std = payment.std()
    scaled_payment = preprocessing.scale(payment)
    std3 += sum(scaled_payment >3)
    std3_index.append(cluster_id_cor[key][scaled_payment >3])
    std3_savings +=np.sum(std*(scaled_payment[scaled_payment >3] - 2))



print('There are %d charge submissions that are 3 standard deviations from other payment amounts in well formed clusters' %std3)

total_payments = np.sum(np.concatenate(list(cluster_payments_cor.values())))

print('Moving outliers from 3 standard deviations away to 2 standard deviations would save %d dollars' %std3_savings)

# finish logging
sys.stdout = old_stdout

log_file.close()