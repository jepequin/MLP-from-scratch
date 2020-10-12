import pandas as pd
import numpy as np

#Define sigmoid function
def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

#Allow sigmoid to be applied to arrays	
vector_sigmoid = np.vectorize(sigmoid)

#Calculate hidden_layer and output values given weight and input values
def forward(weights,feature_vectors):
	hidden_layer = [vector_sigmoid(np.dot(weights[0],feature_vector.transpose())) for feature_vector in feature_vectors]
	output = [vector_sigmoid(np.dot(weights[1],hidden)) for hidden in hidden_layer]
	return hidden_layer, output

#Calculate gradient of error function
def error_gradient(weights,df):
	#Split dataframe into features and classes (scaling features for better sigmoid accuracy)
	feature_vectors = (1/100)*df.iloc[:,:-1].values
	classes = df.iloc[:,-1].values
	hidden, output = forward(weights,feature_vectors)
	coefficients = np.array([(classes[i]-output[i])*output[i]*(1-output[i]) for i in range(len(output))]).transpose()
	matrices = np.array([[hidden[i][j]*(1-hidden[i][j])*weights[1][j]*feature_vectors[i] for j in range(10)] for i in range(len(feature_vectors))]) 
	first_gradient = -(2/len(feature_vectors))*sum((coefficients*matrices.transpose()).transpose())
	second_gradient = -(2/len(feature_vectors))*np.dot(coefficients,hidden)
	return first_gradient, second_gradient

#Update weights
def update_weights(weights,df):
	return [weights[i]-error_gradient(weights,df)[i] for i in range(2)]

#Train the model
def fit(df,epochs=1000):
	#Initialize weights
	weights = [np.random.random((10,8)), np.random.random((1,10))[0]]
	for i in range(epochs):
		weights = update_weights(weights,df)
	return weights

#Make a prediction
def predict(weights,feature_vector):
	probability = forward(weights,[feature_vector])[1][0]
	print(' ')
	print('The predicted probability for this patient having diabetes is : {}'.format(probability))
	print('Note: previous patient actually has diabetes')
	print(' ')
	pass	
	
#Load csv file
df = pd.read_csv('pima-indians-diabetes.csv',header=None)

#Find weights (increase value of epochs for better accuracy)
weights = fit(df)

predict(weights,np.array([10,129,62,36,0,41.2,0.441,38]))