from yahoo_fin import stock_info as si
import pandas as pd
from datetime import datetime
from yahoofinancials import YahooFinancials
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os 

# Define the tickers
tickers = ["MMM", "GS", "NKE", "AXP", "HD", "PG", "AMGN", "HON", "CRM", "AAPL", "INTC", "TRV", "BA", "IBM",
          "UNH", "CAT", "JNJ", "VZ", "CVX", "JPM", "V", "CSCO", "MCD", "WBA", "KO", "MRK", "WMT", "DOW", 
           "MSFT", "DIS"]

tickers_1 = ["^DJI"]

# Get today's date
today = datetime.today().strftime('%Y-%m-%d')

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
fecha_inicio = '2022-01-03'

indice_inicio =  returns_data_sp.index.get_loc(fecha_inicio)
index_values = []
port_values = []
dates_values = []

for i in range(indice_inicio, len(returns_data_sp)-2):
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
        
        price_d_data['rt_'+i] = np.log(price_d_data[i] / price_d_data[i].shift(1))
        returns_d_data = pd.concat([returns_d_data, price_d_data['rt_'+i]], axis=1)
    cov_matrix = returns_data.cov()
    corr_matrix = returns_data.corr()
    dist_matrix = np.sqrt(0.5*(1-corr_matrix))
    dist_matrix = dist_matrix.values
    # Get the number of columns in X
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

    #plt.figure(figsize=(14, 7))
    dendrogram = sch.dendrogram(linkage_matrix, labels=tickers)
    #plt.show()
    leaf_order = leaves_list(linkage_matrix)

    # reorder the columns of the correlation matrix based on the leaf order
    cov_reordered = cov_matrix.iloc[leaf_order, leaf_order]
    corr_reordered = corr_matrix.iloc[leaf_order, leaf_order]
    tickers_reor = cov_reordered.columns.tolist()

    # plot the original and reordered correlation matrices side-by-side
    fig, axs = plt.subplots(ncols=2, figsize=(13, 5))

    # plot the original correlation matrix
    sns.heatmap(corr_matrix, cmap='Blues', ax=axs[0])
    axs[0].set_title('Original Correlation Matrix')

    # plot the reordered correlation matrix
    sns.heatmap(corr_reordered, cmap='Blues', ax=axs[1])
    axs[1].set_title('Reordered Correlation Matrix')

    plt.show()
    indexes = cov_reordered.columns
    weights=pd.Series(1, index=indexes)
    cItem=[indexes]
    while len(cItem)>0:
        cItem=[i[j:k] for i in cItem for j,k in ((0,len(i)//2), (len(i)//2,len(i))) if len(i)>1]
        for i in range(0,len(cItem),2):
            cItem1=cItem[i] # cluster 1
            cItem2=cItem[i+1] # cluster 2 

            left_matrix = cov_reordered.loc[cItem1,cItem1]
            right_matrix = cov_reordered.loc[cItem2,cItem2]

            w1=1./np.diag(left_matrix)
            w1/=w1.sum()
            w2=1./np.diag(right_matrix)
            w2/=w2.sum()
            V1 = w1.T @ left_matrix @ w1
            V2 = w2.T @ right_matrix @ w2

            alpha = 1 - (V1/(V1+V2))

            weights[cItem1] *= alpha
            weights[cItem2] *= 1-alpha
    weights = weights.reindex(returns_data.columns)
    # calculate portfolio returns
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
    weights = weights.rename(index=lambda x: x.replace('rt_', ''))

index_graph = (np.array(index_values) + 1).cumprod() * 1000
port_graph = (np.array(port_values) + 1).cumprod() * 1000

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(dates_values, port_graph, label='Portfolio HRP')
ax.plot(dates_values, index_graph, label='DJI Index')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value')
ax.set_title('Portfolio Value Over Time')
ax.legend(loc='upper left')

df = pd.DataFrame({'Column1': index_graph, 'Column2': port_graph, 'Column3': dates_values})

#df.to_excel('C:\\--\--\--\HRP_Simulation.xlsx', index=False)
