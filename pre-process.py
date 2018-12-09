from datetime import datetime
import os
from pandas import read_csv
from datetime import datetime
import pandas as pd

# Load the dataset
dataset = read_csv("oct_march.csv", index_col=None)

dataset = dataset[['views','subscriber','videoscount','date']].values.tolist()

#print dataset[0]

columns = ['views','subscriber','videoscount','date'];

#txs = pd.read_table('oct_march.csv', sep=',', header=None, names=columns)

txs = pd.read_csv('oct_march.csv', names=columns)

txs = txs[1:]

txs.info() # to get summary statistics
txs.head() # to get a feel for the data

#print txs['date'][1:]

year = lambda x: datetime.strptime(x, "%Y-%m-%d" ).year
txs['year'] = txs['date'].map(year)

day_of_week = lambda x: datetime.strptime(x, "%Y-%m-%d" ).weekday()
txs['day_of_week'] = txs['date'].map(day_of_week)
month = lambda x: datetime.strptime(x, "%Y-%m-%d" ).month
txs['month'] = txs['date'].map(month)
day = lambda x: datetime.strptime(x, "%Y-%m-%d" ).day
txs['day'] = txs['date'].map(day)
# please read docs on how week numbers are calculate
week_number = lambda x: datetime.strptime(x, "%Y-%m-%d" ).strftime('%V')
txs['week_number'] = txs['date'].map(week_number)

seasons = [0,0,1,1,1,2,2,2,3,3,3,0] #dec - feb is winter, then spring, summer, fall etc
season = lambda x: seasons[(datetime.strptime(x, "%Y-%m-%d" ).month-1)]
txs['season'] = txs['date'].map(season)

quarters = [0,0,0,1,1,1,2,2,2,3,3,3] #quarter
quarter = lambda x: quarters[(datetime.strptime(x, "%Y-%m-%d" ).month-1)]
txs['quarter'] = txs['date'].map(quarter)

part_of_months = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2] #quarter
part_of_month = lambda x: part_of_months[(datetime.strptime(x, "%Y-%m-%d" ).day-1)]
txs['part_of_month'] = txs['date'].map(part_of_month)

txs.to_csv('data.csv', index=False)

