
#this is how the data looks like initially. Uncomment if you want to see the initial dataset on its own

from pandas import read_csv
from matplotlib import pyplot

#series = read_csv('petrol_prices.csv', header=0, index_col=0)
#print(series.head())
#series.plot()
#pyplot.show()


from statsmodels.tsa import ar_model 
from statsmodels.tsa.ar_model import AutoReg, AutoRegResults
from sklearn.metrics import mean_squared_error
import numpy
from math import sqrt

# create a difference transform of the dataset
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

# Make a prediction give regression coefficients and observations
def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat

#read our petrol data in
series = read_csv('petrol_prices.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

# split dataset
X = difference(series.values)
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:]

# train autoregression
window = 10
model = ar_model.AutoReg(train, lags=6)
model_fit = model.fit()
coef = model_fit.params

# walk forward over time steps in test
history = [train[i] for i in range(len(train))]
predictions = list()
for t in range(len(test)):
	yhat = predict(coef, history)
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
rmse = sqrt(mean_squared_error(test, predictions))

#the mean sq. error
print('Test RMSE: %.3f' % rmse)

# plot. Blue - real, green - predicted.
pyplot.plot(test)
pyplot.plot(predictions, color='green')
pyplot.show()

# fit an AR model and save the whole model to file

# create a difference transform of the dataset
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

# load dataset

#series = read_csv('petrol_prices.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

X = difference(series.values)
# fit model
model = AutoReg(X, lags=6)
model_fit = model.fit()
# save model to file
model_fit.save('petrol_model.pkl')
# save the differenced dataset
numpy.save('petrol_data.npy', X)
# save the last ob
numpy.save('petrol_obs.npy', [series.values[-1]])

# load AR model from file and make a one-step prediction

# load model
model = AutoRegResults.load('petrol_model.pkl')
data = numpy.load('petrol_data.npy')
last_ob = numpy.load('petrol_obs.npy')
# make prediction
predictions = model.predict(start=len(data), end=len(data))
# transform prediction
yhat = predictions[0] + last_ob[0]
print('Prediction for next week: %f' % yhat)


# # update the data for the manual model with a new observation once available
# import numpy
# # get real observation
# observation = 48
# # update and save differenced observation
# lag = numpy.load('man_data.npy')
# last_ob = numpy.load('man_obs.npy')
# diffed = observation - last_ob[0]
# lag = numpy.append(lag[1:], [diffed], axis=0)
# numpy.save('man_data.npy', lag)
# # update and save real observation
# last_ob[0] = observation
# numpy.save('man_obs.npy', last_ob)