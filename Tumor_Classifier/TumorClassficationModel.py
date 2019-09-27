import sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Load data

data = load_breast_cancer()

#Important dictionary keys: 
# classification label names => 'target_names'
#actual labels => 'target'
#attribute/feature names => 'feature_names'
#attributes => 'data'

#Organize data 

label_names = data['target_names'] #
labels = data['target']
feature_names = data['feature_names']
features = data['data']

#Looking at the data

""" print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0]) """

#Split the data. 33% used for test set

train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

#Initialize Classifier

gnb = GaussianNB()

#Train the Classifier

model = gnb.fit(train, train_labels)

 #Make predictions 

preds = gnb.predict(test)

#predict() returns binary array 
'''print(preds)'''

#Evaluate Accuracy 

print(accuracy_score(test_labels, preds))
