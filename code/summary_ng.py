import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadr
import argparse
import sklearn.metrics as metrics

parser = argparse.ArgumentParser()
parser.add_argument("--n", help="Sample size n",
                    type=int)
parser.add_argument("--p", help="Number of covariates",
                    type=int)
parser.add_argument("--J", help="Number of clusters",
                    type=int)
parser.add_argument("--ind", help="repetition index",
                    type=int)

args = parser.parse_args()
n = args.n
p = args.p
n_clus = args.J
ind = args.ind

burn_in = 2000

dir_current = "n%i_p%i_clus%i_ng/n%i_p%i_clus%i_ind%i_ng/" %(n,p,n_clus,n,p,n_clus,ind)

# =============================================================================
# Load results from MCMC
# =============================================================================
with open(dir_current + "result_n%i_p%i_cluster%i.pickle" %(n,p,n_clus),"rb") as f:
    sil_score,cluster_index,beta_true,clus_beta = pickle.load(f)
cluster_index = cluster_index - 1

# =============================================================================
# Load clustering result obtained from R
# =============================================================================
result = pyreadr.read_r(dir_current + "cluster.Rdata")

cluster_rastelli = np.array(result['cluster']['cluster'].tolist())
cluster_dahl = np.array(result['cluster_ls']['cluster_ls'].tolist())
ari_rastelli = np.array(result['ari_rastelli']['ari_rastelli'].tolist())
ari_dahl = np.array(result['ari_dahl']['ari_dahl'].tolist())
rand_rastelli = np.array(result['rand_rastelli']['rand_rastelli'].tolist())
rand_dahl = np.array(result['rand_dahl']['rand_dahl'].tolist())

clus_to_save = (cluster_rastelli, cluster_dahl, ari_rastelli, ari_dahl, rand_rastelli, rand_dahl)
with open(dir_current + "cluster_n%i_p%i_cluster%i.pickle" %(n,p,n_clus),"wb") as f:
    pickle.dump(clus_to_save, f)

# =============================================================================
# check variable selection result
# =============================================================================
auc_tmp = []
for i in range(n):
    tmp_beta_true = beta_true[cluster_index[i]-1,:]
    tmp_clus_beta = clus_beta[cluster_rastelli[i]][0][clus_beta[cluster_rastelli[i]][0][:,0] != 0,:] # posterior samples corresponding to sample i
    var_beta = np.var(tmp_clus_beta, axis = 0)
    std_beta = np.sqrt(var_beta)
    exclusion_rate = []
    for l in range(p):
        exclusion_rate.append( np.sum( np.where((tmp_clus_beta[:,l] < std_beta[l]) & (tmp_clus_beta[:,l] > -std_beta[l]),1,0) ) / tmp_clus_beta[:,l].shape[0] )
    exclusion_rate = np.array(exclusion_rate)
    inclusion_rate = 1 - exclusion_rate # predicted probabilities
    inclusion_true = (tmp_beta_true != 0).astype(int) # true label    
    auc_tmp.append(metrics.roc_auc_score(inclusion_true, inclusion_rate)) # auc score for each sample i

auc_avg = np.mean(auc_tmp)

# =============================================================================
# check clustering result
# =============================================================================
unique_clus_ind = result['cluster_ind_sort']['cluster_ind_sort'].astype(int).tolist()

### check each cluster's beta distirbution and median
median_list = []
for unique_id in unique_clus_ind:
    tmp_clus_beta = clus_beta[unique_id][0][clus_beta[unique_id][0][:,0] != 0,:] # drop values not really recorded
    median = np.median(tmp_clus_beta, axis = 0) # median of posterior \beta for the cluster
    median_list.append(np.round(median,2))
    print(median_list[-1])

mse_overall = 0
for i in range(n):
    tmp_id = cluster_rastelli[i]
    tmp_beta_median = median_list[np.where(unique_clus_ind == tmp_id)[0][0]] ### posterior beta median corresponding to this data sample
    tmp_beta_true = beta_true[cluster_index[i],:]
    mse_overall = mse_overall + np.sum((tmp_beta_median - tmp_beta_true) ** 2)
mse_overall = mse_overall / (n * p)    

output = ""
output = output + "Average AUC: %.2f \n" % auc_avg
output = output + "Separation Index(average Silhouette score): %.2f \n" % sil_score
output = output + "Adjusted Rand Index of Rastelli and Dahl: %.2f, %.2f \n" % (ari_rastelli,ari_dahl)
output = output + "Rand Index of Rastelli and Dahl: %.2f, %.2f \n" % (rand_rastelli,rand_dahl)
output = output + "Estimated number of clusters: %d \n" % len(unique_clus_ind) 
output = output + "Overall MSE: %.4f \n" % mse_overall
for idx,tmp_median in enumerate(median_list):
    output = output + "Beta medians of cluster %d: " % (idx+1) + " & ".join( [str(item) for item in tmp_median] ) + "\n"

ase_to_save = (mse_overall,len(unique_clus_ind),sil_score)
with open(dir_current + "ase_n%i_p%i_cluster%i.pickle" %(n,p,n_clus),"wb") as f:
    pickle.dump(clus_to_save, f)

with open(dir_current + "Output_estimation.txt", "w") as text_file:
    text_file.write(output)

######### Delete result...pickle
os.remove(dir_current + "result_n%i_p%i_cluster%i.pickle" %(n,p,n_clus))

