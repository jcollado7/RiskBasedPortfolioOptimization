from yahoo_fin import stock_info as si
import pandas as pd
from datetime import datetime
from yahoofinancials import YahooFinancials
import numpy as np
from scipy.stats import norm
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, cut_tree, cophenet, fcluster
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist
import random
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

# Define the tickers
tickers = ["MMM", "GS", "NKE", "AXP", "HD", "PG", "AMGN", "HON", "CRM", "AAPL", "INTC", "TRV", "BA", "IBM",
          "UNH", "CAT", "JNJ", "VZ", "CVX", "JPM", "V", "CSCO", "MCD", "WBA", "KO", "MRK", "WMT", "DOW", 
           "MSFT", "DIS"]

# Get today's date
today = datetime.today().strftime('%Y-%m-%d')

# Use YahooFinancials to retrieve historic price data for the tickers
price_data = pd.DataFrame()
for ticker in tickers:
    yahoo_financials = YahooFinancials(ticker)
    historic_price_data = yahoo_financials.get_historical_price_data(start_date='2020-11-01', end_date=today, time_interval='daily')
    prices = historic_price_data[ticker]['prices']
    prices_df = pd.DataFrame(prices)
    prices_df = prices_df.set_index('formatted_date')
    prices_df.index.name = 'date'
    prices_df = prices_df['adjclose'].rename(ticker)
    price_data = pd.concat([price_data, prices_df], axis=1)

# Sort the price data by date
price_data = price_data.sort_index()
returns_data = pd.DataFrame()
for i in tickers:
    price_data['rt_'+i] = np.log(price_data[i] / price_data[i].shift(1))
    returns_data = pd.concat([returns_data, price_data['rt_'+i]], axis=1)
cov_matrix = returns_data.cov()
corr_matrix = returns_data.corr()
dist_matrix = np.sqrt(0.5*(1-corr_matrix))
dist_matrix = dist_matrix.values
k = dist_matrix.shape[1]

# Initialize the output matrix D
D = np.zeros((k, k))

# Compute the distance between each pair of columns
for i in range(k):
    for j in range(k):
        d = 0
        for n in range(k):
            d += (dist_matrix[n, i] - dist_matrix[n, j])**2
        D[i, j] = np.sqrt(d)

# Print the output matrix D
distance = pd.DataFrame(D, columns=tickers, index=tickers)
linkage_matrix = sch.linkage(distance, method='ward')
plt.figure(figsize=(14, 7))
dendrogram = sch.dendrogram(linkage_matrix, labels=tickers)

#------------------------------------------ Optimal Number of CLusters----------------------------------------------
# Fill missing values with zeros
returns_data_filled = returns_data.fillna(0)

# Define the number of reference datasets to generate
num_reference_datasets = 10

# Generate reference datasets (random data) with the same dimensions as the original data
reference_datasets = [make_blobs(n_samples=returns_data_filled.shape[0], centers=returns_data_filled.shape[1])[0] for _ in range(num_reference_datasets)]

# Calculate the within-cluster dispersion for the original data
def within_cluster_dispersion(data, clusters):
    centers = np.array([data[clusters == i].mean(axis=0) for i in np.unique(clusters)])
    return np.sum((data - centers[clusters]) ** 2)

# Calculate the gap statistic for a given number of clusters
def gap_statistic(data, num_clusters, num_reference_datasets):
    clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    clusters = clustering.fit_predict(data)
    within_dispersion = within_cluster_dispersion(data, clusters)
    
    # Calculate the gap for the original data
    gap = np.log(within_dispersion)
    
    # Calculate the gap for reference datasets
    reference_gaps = []
    for ref_data in reference_datasets:
        ref_clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
        ref_clusters = ref_clustering.fit_predict(ref_data)
        ref_within_dispersion = within_cluster_dispersion(ref_data, ref_clusters)
        reference_gaps.append(np.log(ref_within_dispersion))
    
    # Calculate the reference gap
    reference_gap = np.mean(reference_gaps)
    
    # Calculate the standard deviation of reference gaps
    reference_gap_std = np.std(reference_gaps)
    
    # Calculate the gap statistic
    gap_stat = reference_gap - gap
    
    # Calculate the standard error
    std_err = reference_gap_std * np.sqrt(1 + 1 / num_reference_datasets)
    
    return gap_stat, std_err

# Define a range of cluster numbers to try
num_clusters_range = range(2, len(tickers) + 1)
print(num_clusters_range)
# Initialize variables to store the optimal number of clusters and the corresponding gap statistic
optimal_num_clusters = 10
max_gap_stat = -np.inf

# Calculate the gap statistic for each number of clusters
for num_clusters in num_clusters_range:
    gap_stat, std_err = gap_statistic(returns_data_filled, num_clusters, num_reference_datasets)
    
    # Compare the mean of gap_stat with max_gap_stat
    if gap_stat.mean() > max_gap_stat:
        max_gap_stat = gap_stat.mean()
        optimal_num_clusters = num_clusters

print("Optimal number of clusters:", optimal_num_clusters)

#rd_sm = np.random.random_sample(size=D.shape)
#random_sample = pd.DataFrame(rd_sm, columns=tickers, index=tickers)
#linkage_random = sch.linkage(random_sample, method='ward')
#Kmax = 10
#W = []
#gap_stats = np.zeros((Kmax,))
#for k in range(1, optimal_num_clusters+1):
#    labels = fcluster(linkage_matrix, k, criterion='maxclust')
#    labels_rm = fcluster(linkage_random, k, criterion='maxclust')
#    cluster_distances = []
#    cluster_distances_rm = []
#    for i,j in zip(np.unique(labels), np.unique(labels_rm)):
#        cluster_points = np.where(labels == i)[0]
#        cluster_distances.append(D[np.ix_(cluster_points, cluster_points)].sum()/ (2 * len(cluster_points)))
#        cluster_points_rm = np.where(labels_rm == j)[0]
#        cluster_distances_rm.append(rd_sm[np.ix_(cluster_points_rm, cluster_points_rm)].sum()/ (2 * len(cluster_points_rm)))
#    gap_stats[k-1] = np.log(sum(cluster_distances_rm)) - np.log(sum(cluster_distances))
#optimal_k = np.argmax(gap_stats)
#print("Optimal number of clusters:", optimal_k)

#-----------------------------------------------------------------------------------

cov_matrix = cov_matrix.rename(index=lambda x: x.replace('rt_', ''))
cov_matrix = cov_matrix.rename(columns=lambda x: x.replace('rt_', ''))
std_data = returns_data.std()
confidence_level = 0.99
stock_VaR = -(std_data * norm.ppf(1 - confidence_level))

# Calculate the CVaR (Conditional Value at Risk) for each stock.
stock_cVaR = {}
for stock in returns_data.columns:
    stock_return = returns_data[stock]
    beyond_var = stock_return <= -stock_VaR[stock]
    cVaR = -stock_return[beyond_var].mean()
    stock_cVaR[stock] = cVaR

stock_cVaR = pd.Series(stock_cVaR)
weights_dict = {stock: 1 for stock in tickers}
for i in range(optimal_num_clusters-1):
    if i > 0:
        t = linkage_matrix[-i-1,-2]-0.000001
        clusters = fcluster(linkage_matrix, t=t, criterion='distance')
        mask = np.array([clusters == val for val in np.unique(clusters)])
        clusters_list_2 = [np.array(tickers)[m].tolist() for m in mask]
        different_clusters = []
        for cluster2 in clusters_list_2:
            match_found = False
            for i, cluster1 in enumerate(clusters_list_1):
                if cluster1 == cluster2:
                    match_found = True
                    break
            if not match_found:
                different_clusters.append(cluster2)
        clusters_list_1 = clusters_list_2
        weights_clusters = []        
        sum_of_lists = [stock_cVaR.loc[['rt_' + asset for asset in assets_list]].sum() for assets_list in different_clusters]
        total_sum_of_inverse = sum(1 / sum_of_list for sum_of_list in sum_of_lists)
        weights_clusters = [(1 / sum_of_list) / total_sum_of_inverse for sum_of_list in sum_of_lists]
        print(different_clusters)
        print(weights_clusters)
        for i, sublist in enumerate(different_clusters):
            weight_cluster = weights_clusters[i]
            for stock in sublist:
                weights_dict[stock] *= weight_cluster 
                
    else:
        t = linkage_matrix[-i-1,-2]-0.001
        clusters = fcluster(linkage_matrix, t=t, criterion='distance')
        mask = np.array([clusters == val for val in np.unique(clusters)])
        clusters_list_1 = [np.array(tickers)[m].tolist() for m in mask]
        weights_clusters = []
        sum_of_lists = [stock_cVaR.loc[['rt_' + asset for asset in assets_list]].sum() for assets_list in clusters_list_1]
        total_sum_of_inverse = sum(1 / sum_of_list for sum_of_list in sum_of_lists)
        weights_clusters = [(1 / sum_of_list) / total_sum_of_inverse for sum_of_list in sum_of_lists]
        print(clusters_list_1)
        print(weights_clusters)
        for i, sublist in enumerate(clusters_list_1):
            weight_cluster = weights_clusters[i]
            for stock in sublist:
                weights_dict[stock] *= weight_cluster 
clusters_final = clusters_list_1
for sublist in clusters_final:
    cov_cluster = cov_matrix.loc[sublist, sublist]
    for stock in sublist:
        weight_stock = (1 / (cov_cluster.loc[stock, stock]) / sum(1 / (cov_cluster.loc[stock, stock]) for stock in sublist))
        weights_dict[stock] *= weight_stock
        
weights = pd.Series(weights_dict)
print("Final weights \n", weights)
