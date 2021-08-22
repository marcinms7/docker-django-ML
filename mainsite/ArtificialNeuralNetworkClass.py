#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 19:25:57 2021

@author: marcinswierczewski

The module applies the Neural Network for binary clasifcation.
For detailed specification of the model, please visit Model Explanation
on the website.
"""




import numpy as np
np.random.seed(66)



class ArtificialNeuralNetwork:
    
    def __init__(self, train_data, pred_data, learning_rate = 0.01,
                 epochs = 50, hidden_layers = 2):
        '''
        

        Parameters
        ----------
        train_data : csv
            Dataset that will be split into training and test data.
        pred_data : csv
            Data that will have applied neural network to, and appended
            the classification results to the last column.
        learning_rate : Float, optional
            Learning rate of the model. The default is 0.01.
        epochs : Int, optional
            Number of iterations. The default is 50.
        hidden_layers : Int, optional
            Hiden layers. For detailed explanation please see
            /modelexplanation page. The default is 2.

        Returns
        -------
        Please call predict_data_and_save() to train the neural network 
        on the portion of train data and assess accuracy score on a remaining 
        portion of the data. Please call predict_data_and_save to run trained
        neural network on the pred_data df, and save/append results to the last
        column of that df. 

        '''
        self.train_data = train_data
        self.test_data = None
        self.actual_from_test = None
        self.pred_data = pred_data
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.file_split()
        self.network = []
        
    def df_to_list(self, dataset):
        # This function converts any dataset into a list of floats, with
        # the last element of the list being integer.
        return list(list(x) for x in zip(*(dataset[x].values.tolist() for x in dataset.columns)))

    def file_split(self):
        # This function splits the data into a training and test set, for 
        # test setting target to None, to avoid data leakage.
        split = int(self.train_data.shape[0] * 0.8)
        dataset = self.train_data.copy()
        self.train_data = dataset[:split]
        self.test_data = dataset[split:]
        del dataset
        # keeping actual data from test set for later AUC and comparisons 
        self.actual_from_test = list(self.test_data[self.test_data.columns[-1]])
        self.test_data[self.test_data.columns[-1]] = None
        
        self.train_data = self.df_to_list(self.train_data)
        self.test_data = self.df_to_list(self.test_data)


    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        # derivative of sigmoid for back propagation
            return self.sigmoid(x)*(1-self.sigmoid(x))
    
    
    def initiate(self, inputs,outputs):
        # initiating neural network 
        self.hidden_layer= []
        # adding hidden layers
        for i in range(self.hidden_layers):
            ltemp = []
            # the below sometimes recomended as inputs + 1, additional one for bias
            for i in range(inputs +1):
                ltemp.append(np.random.random())
            dtemp = {'weights' : ltemp}
            self.hidden_layer.append(dtemp)
        self.network.append(self.hidden_layer)
        output = []
        # adding output layer
        for i in range(outputs):
            ltemp = []
            # the below sometimes recomended as hiddenlayers + 1, additional one for bias
            for i in range(self.hidden_layers +1):
                ltemp.append(np.random.random())
            dtemp = {'weights' : ltemp}
            output.append(dtemp)
        self.network.append(output)
     
    
     
    # forward propagation
    def activation(self, features, weights, bias = 1):
        # that was changed from dot product of weights and feature matrices
    	activation = weights[-1]
    	for i in range(len(weights)-2):
    		activation += (weights[i] * features[i]) + bias
    	return activation 
    
        
    def forward_propagate(self, data):
        inputs = data
        for layer in self.network:
            new = []
            for neuron in layer:
                activ = self.activation(inputs, neuron['weights'])
                neuron['output'] = self.sigmoid(activ)
                new.append(neuron['output'])
            inputs = new
        return inputs
        
    
    # backward propagation
    def backward_propagate(self, expected):
        # using 'reversed' iterator to back-propagate through nn
        for i in reversed(range(len(self.network))):
            # iterating for each layer, setting layer to i
            layer = self.network[i]
            errors = []
            # if not last layer (first propagated)
            # so basically for all hidden layers 
            if i != len(self.network)-1:
                # iterating through each element of the layer
                for j in range(len(layer)):
                    error = 0.0
                    # iterating through each neuron of the layer
                    for neuron in self.network[i + 1]:
                        # updating errors and appending to errors list 
                        # hidden neuron number j is also the index of 
                        # the neuron’s weight in the output layer neuron['weights'][j]
                        error+= (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                # for first iteration - initiating errors
                # same loop as above
                for j in range(len(layer)):
                    # appending error for each neuron of the current layer
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                # iterating through all neurons again, and
                # updating by slope (derivative) of each output
                neuron = layer[j]
                # 'delta' is the error signal
                neuron['delta'] = errors[j] * self.sigmoid_derivative(neuron['output'])
        
     
    def gradient_descent(self, data):
        # standard gradient function
        for i in range(len(self.network)):
            inputs = data[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    # neuron weights are updated by learning rate , delta  and input
                    neuron['weights'][j] += self.learning_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += self.learning_rate * neuron['delta']
     
    def train(self, data ):
        for i in range(self.epochs):
            sumerror = 0
            for row in data:
                output = self.forward_propagate(row)
                expected = [0 for i in range(len(set([lastrow[-1] for lastrow in data])))]
                expected[row[-1]] = 1
                sumerror += sum([(expected[i]-output[i])**2 for i in range(len(expected))])
                self.backward_propagate(expected)
                self.gradient_descent(row)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (i, self.learning_rate, sumerror))
                
    def predict(self, row):
        # it predicts classification on already trained Network
        return (self.forward_propagate(row)).index(max(self.forward_propagate(row)))
            
    def predict_function(self, data):
        # function applying the above predict() and returning predictions array
        predictions = []
        for row in data:
            prediction = self.predict(row)
            predictions.append(prediction)
        return predictions

    
    def accuracy(self, actual, pred):
        # simple calculation of accuracy 
        good = 0
        count_total = len(actual)
        for i in range (count_total):
            if actual[i] == pred[i]:
                good += 1
        return good / count_total
    
    def algorithm(self):
        # initiating Neural Network with appropariate dimension, training it 
        # on the train data and testing with test data.
        n_inputs = len(self.train_data[0])
        n_outputs = len(set([row[-1] for row in self.train_data]))
    
        self.initiate(n_inputs, n_outputs)
        self.train(self.train_data)
        return self.predict_function(self.test_data)
    
    def train_and_test(self):
        # applying above train/test algorithm function, returning accuracy score
        predicted = self.algorithm()
        actual = self.actual_from_test
        print(predicted, actual)
        return self.accuracy(actual, predicted)
        
        
    def predict_data_and_save(self):
        # applying trained Neural Network on the held out predictor data, 
        # that had empty Target column (last column) and saving the predicted
        # result by appending it to the last column of that dataset
        assert self.network != []
        df = self.pred_data.copy()
        df = self.df_to_list(df)
        pred = self.predict_function(df)
        self.pred_data[self.pred_data.columns[-1]] = pred
        return self.pred_data


