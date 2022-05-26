# importing libraries
import numpy as np
import pandas as pd

# plot libraries
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (16, 6)

#dataset reading
df = pd.read_csv("https://pycourse.s3.amazonaws.com/bike-sharing.csv")

# date time conversion
df['datetime'] = pd.to_datetime(df['datetime'])

# categorical variables
df['season'] = df['season'].astype('int')
df['is_holiday'] = df['is_holiday'].astype('int')
df['weekday'] = df['weekday'].astype('int')
df['weather_condition'] = df['weather_condition'].astype('int')
df['is_workingday'] = df['is_workingday'].astype('int')
df['month'] = df['month'].astype('int')
df['year'] = df['year'].astype('int')
df['hour'] = df['hour'].astype('int')

#total count plot
df.plot(x='datetime', y='total_count');

# hourly tendencies
_, ax = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))

sns.pointplot(data=df, x='hour', y='total_count', hue='season', ax=ax[0])
ax[0].set_title("Hourly bike rentals by season")
ax[0].grid()

sns.pointplot(data=df, x='hour', y='total_count', hue='weekday', ax=ax[1])
ax[1].set_title("Hourly bike rentals by day of the week")
ax[1].grid();

# distribution by month
fig, ax = plt.subplots()
sns.barplot(data=df, x="month", y="total_count")
ax.set_title("Monthly Distribution");

# distribution by season
fig, ax = plt.subplots()
sns.barplot(data=df, x="season", y="total_count")
ax.set_title("Distribution by season");

# distribution by year
fig, ax = plt.subplots()
sns.barplot(data=df, x="year", y="total_count", estimator=sum, ci=None)
ax.set_title("Distribution by year");

# Simplified predictive rental model
#
# data preprocessing: removing unnecessary columns
df.drop(['rec_id',
         'casual',
         'registered',
         'atemp',
         'year',
         'hour'],
        axis=1,
        inplace=True)

# group by datetime
df = df.groupby('datetime', as_index=False).mean()        

# grouped data plot
df.plot(x='datetime', y='total_count');

# generating the lagged series total_count: lag1
df['total_count_lag1'] = np.r_[df.iloc[0, -1], df.iloc[:-1, -1]]

# ajusting dtypes
df['season'] = df['season'].astype('int')
df['month'] = df['month'].astype('int')
df['is_holiday'] = df['is_holiday'].astype('int')
df['weekday'] = df['weekday'].astype('int')
df['is_workingday'] = df['is_workingday'].astype('int')

# division of train (90%) and test (10%)
n, p = df.shape[0], 0.9

df_train = df.iloc[:int(n*p), 1:]
dtime_train = df.iloc[:int(n*p), 0]

df_test = df.iloc[int(n*p):, 1:]
dtime_test = df.iloc[int(n*p):, 0]

# extraction of x_train, x_test, y_train, y_test
x_train = df_train.drop('total_count', axis=1)
y_train = df_train['total_count']

x_test = df_test.drop('total_count', axis=1)
y_test = df_test['total_count']

# model
from sklearn.ensemble import RandomForestRegressor

# fit model
model = RandomForestRegressor(random_state=0)
model.fit(x_train, y_train)

# prediction
y_pred = model.predict(x_test)

# confidence interval
n_steps = 3
ts_pred = pd.DataFrame(y_pred)
smooth_path    = ts_pred.rolling(n_steps).mean()
path_deviation = 1.96 * ts_pred.rolling(n_steps).std()

under_line = (smooth_path-path_deviation)[0]
over_line  = (smooth_path+path_deviation)[0]

# plot
plt.plot(dtime_test, y_pred, linewidth=2, label='Prediction')
plt.fill_between(dtime_test, under_line, over_line, color='b', alpha=.15)
plt.plot(dtime_test, y_test, label='Real')
plt.xticks(dtime_test.iloc[np.arange(dtime_test.size, step= 10).astype(int)])
plt.legend()
plt.grid()
plt.title("Prediction of the number of daily bicycle rentals")
plt.show()

# importance of features
fp = model.feature_importances_
n = 5
i = np.argsort(fp)[-n:]
cols = x_train.columns
plt.barh(cols[i], fp[i])
plt.grid()
plt.title(f"{n} most important features")
plt.xlabel("Relative Importance")
plt.ylabel("Feature")
plt.show()