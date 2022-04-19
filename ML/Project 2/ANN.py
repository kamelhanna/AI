#Import Lib.
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,confusion_matrix

#Read dataset and preprocess it
df = pd.read_csv('Dataset_spine.csv')
df = df.drop(['Unnamed: 13'], axis=1)
df.head()

df.describe()

#assign input and output 
y = df['Class_att']
x = df.drop(['Class_att'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25,random_state=27)

x_train
x_test
y_train
y_test 


'''
MLPClassifierModel = MLPClassifier(activation='tanh', # can be also identity , logistic , relu
                                   solver='lbfgs',  # can be also sgd , adam
                                   learning_rate='constant', # can be also invscaling , adaptive
                                   early_stopping= False,
                                   alpha=0.0001 ,hidden_layer_sizes=(100, 3),random_state=33)
'''




clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=500, alpha=0.0001,solver='adam', verbose=10,  random_state=21,tol=0.000000001)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

accuracy_score(y_test, y_pred)


cm = confusion_matrix(y_test, y_pred)
cm

sns.heatmap(cm, center=True)
plt.show()


