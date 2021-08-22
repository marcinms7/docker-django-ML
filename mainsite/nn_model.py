import pandas as pd 
from .ArtificialNeuralNetworkClass import ArtificialNeuralNetwork
# Create your tests here.
PATH = './media/'
def extract(filename, epochs, hidden_layers_no):
    '''
    The function extracts the uploaded csv, deals with converting data types
    and splits it between training/test dataframe and the dataframe that
    will have the predicted data appended. 
    
    As states in the website's instruction, wherever the row has 'target'
    column empty, it will be set aside for prediction 
    and have the class appended to it.
    
    Parameters
    ----------
    filename : csv
        CSV passed to the function. 
    epochs : Int
        Iterations of algorithm.
    hidden_layers_no : Int
        Number of hidden layers.

    Returns
    -------
    auc : Float
        Accuracy metric, from 0 to 1.
    bool
        Flag whether there is any dataset to have class date to get appended to.

    '''
    df = pd.read_csv(PATH + filename)

    try:
	    del df['Unnamed: 0']
    except:
	    pass

    dftarget = df.iloc[:,-1]
    dfvariables = df.iloc[:,:-1]


    dfvariables = dfvariables.fillna(0)

    for i in dfvariables.columns:
        if dfvariables[i].dtype == 'int64' or dfvariables[i].dtype == 'float64':
            continue
        else:
            try:
                dfvariables[i] = dfvariables[i].astype(float)
            except:
                del dfvariables[i]


    del df

    dfvariables.insert(dfvariables.shape[1], 'Target', dftarget)
    del dftarget

    dftopredict = dfvariables[dfvariables['Target'].isnull() == True]
    dfvariables = dfvariables[dfvariables['Target'].isnull() == False]
    dfvariables['Target'] = dfvariables['Target'].astype(int)

    model = ArtificialNeuralNetwork(dfvariables, dftopredict, learning_rate = 0.01,
                 epochs = epochs, hidden_layers = hidden_layers_no)

    auc = model.train_and_test()

    if len(dftopredict) > 0:
        dfpredicted = model.predict_data_and_save()


        dfpredicted.to_csv(PATH + 'test.csv')
        return auc, True
    else:
        return auc, False







