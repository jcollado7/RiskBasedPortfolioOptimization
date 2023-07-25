from yahoo_fin import stock_info as si
import pandas as pd
from datetime import datetime
from yahoofinancials import YahooFinancials
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, fcluster, cut_tree, cophenet
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch
import random
import os 

# Define the tickers
tickers = ["MMM", "GS", "NKE", "AXP", "HD", "PG", "AMGN", "HON", "CRM", "AAPL", "INTC", "TRV", "BA", "IBM",
          "UNH", "CAT", "JNJ", "VZ", "CVX", "JPM", "V", "CSCO", "MCD", "WBA", "KO", "MRK", "WMT", "DOW", 
           "MSFT", "DIS"]

# Get today's date
today = datetime.today().strftime('%Y-%m-%d')

tickers_1 = ["^DJI"]

# Use YahooFinancials to retrieve historic price data for the tickers
price_data = pd.DataFrame()
for ticker in tickers_1:
    yahoo_financials = YahooFinancials(ticker)
    historic_price_data = yahoo_financials.get_historical_price_data(start_date='2020-11-01', end_date=today, time_interval='daily')
    prices = historic_price_data[ticker]['prices']
    prices_df = pd.DataFrame(prices)
    prices_df = prices_df.set_index('formatted_date')
    prices_df.index.name = 'date'
    prices_df = prices_df['adjclose'].rename(ticker)
    price_data_sp = pd.concat([price_data, prices_df], axis=1)
# Sort the price data by date
price_data_sp = price_data_sp.sort_index()
returns_data_sp = pd.DataFrame()
for i in tickers_1:
    price_data_sp['rt_'+i] = np.log(price_data_sp[i] / price_data_sp[i].shift(1))
    returns_data_sp = pd.concat([returns_data_sp, price_data_sp['rt_'+i]], axis=1)
    
# Get today's date
today = datetime.today().date()
start_date = '2022-01-03'

start_index =  returns_data_sp.index.get_loc(start_date)
index_values = []
port_values = []
dates_values = []
for i in range(start_index, len(returns_data_sp)-2):
    specific_index = i + 1
    # Use YahooFinancials to retrieve historic price data for the tickers
    price_data = pd.DataFrame()
    price_d_data = pd.DataFrame()
    current_date = returns_data_sp.index[specific_index]
    previous_day = returns_data_sp.index[specific_index-1]
    next_day = returns_data_sp.index[specific_index+1]
    for ticker in tickers:
        yahoo_financials = YahooFinancials(ticker)
        historic_price_data = yahoo_financials.get_historical_price_data(start_date='2020-11-01', end_date=current_date, time_interval='daily')
        prices = historic_price_data[ticker]['prices']
        prices_df = pd.DataFrame(prices)
        prices_df = prices_df.set_index('formatted_date')
        prices_df.index.name = 'date'
        prices_df = prices_df['adjclose'].rename(ticker)
        price_data = pd.concat([price_data, prices_df], axis=1)
        
        day_price_data = yahoo_financials.get_historical_price_data(start_date=previous_day, end_date=next_day, time_interval='daily')
        prices_d = day_price_data[ticker]['prices']
        prices_d_df = pd.DataFrame(prices_d)
        prices_d_df = prices_d_df.set_index('formatted_date')
        prices_d_df.index.name = 'date'
        prices_d_df = prices_d_df['adjclose'].rename(ticker)
        price_d_data = pd.concat([price_d_data, prices_d_df], axis=1)

    # Sort the price data by date
    price_data = price_data.sort_index()
    price_d_data = price_d_data.sort_index()
    returns_data = pd.DataFrame()
    returns_d_data = pd.DataFrame()
    for i in tickers:
        price_data['rt_'+i] = np.log(price_data[i] / price_data[i].shift(1))
        returns_data = pd.concat([returns_data, price_data['rt_'+i]], axis=1)
        
        price_d_data[i] = np.log(price_d_data[i] / price_d_data[i].shift(1))
        returns_d_data = pd.concat([returns_d_data, price_d_data[i]], axis=1)
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
    dendrogram = sch.dendrogram(linkage_matrix, labels=tickers)

    #rd_sm = np.random.random_sample(size=D.shape)

    #random_sample = pd.DataFrame(rd_sm, columns=tickers, index=tickers)
    #linkage_random = sch.linkage(random_sample, method='ward')

    optimal_k = 2
    cov_matrix = cov_matrix.rename(index=lambda x: x.replace('rt_', ''))
    cov_matrix = cov_matrix.rename(columns=lambda x: x.replace('rt_', ''))
    std_data = returns_data.std()
    confidence_level = 0.99
    stock_VaR = -(std_data * norm.ppf(1 - confidence_level))

    stock_cVaR = {}
    for stock in returns_data.columns:
        stock_return = returns_data[stock]
        beyond_var = stock_return <= -stock_VaR[stock]
        cVaR = -stock_return[beyond_var].mean()
        stock_cVaR[stock] = cVaR

    stock_cVaR = pd.Series(stock_cVaR)
    weights_dict = {stock: 1 for stock in tickers}
    for i in range(optimal_k-1):
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
    portfolio_returns = (weights * returns_d_data).sum(axis=1)
    index_value = returns_data_sp.iloc[specific_index, 0]
    date_value = returns_data_sp.index[specific_index]
    port_value = portfolio_returns.iloc[-1]
    date_port_value = portfolio_returns.index[-1]
    print(f"Date:{date_value} ; Index Return:{index_value}")
    print(f"Date:{date_port_value} ; Portfolio Return:{port_value}")
    portfolio_values = (portfolio_returns + 1).cumprod() * 1000
    portfolio_spy = (returns_data_sp + 1).cumprod() * 1000
    dates_values.append(date_value)
    index_values.append(index_value)
    port_values.append(port_value)
    
index_graph = (np.array(index_values) + 1).cumprod() * 1000
port_graph = (np.array(port_values) + 1).cumprod() * 1000

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(dates_values, port_graph, label='Portfolio HERC')
ax.plot(dates_values, index_graph, label='DJI Index')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value')
ax.set_title('Portfolio Value Over Time')
ax.legend(loc='upper left')

df = pd.DataFrame({'Column1': index_graph, 'Column2': port_graph, 'Column3': dates_values})
#df.to_excel('C:\\--\--\--\HERC_CVaR_Sim.xlsx', index=False)
