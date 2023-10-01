import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metric import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(r'C:/Users/User/Desktop/Everything/iris/iris.data')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

score = accuracy_score(y_test,y_pred)

pickle_out = open('model_iris.pkl','wb')
pickle.dump(classifier, pickle_out)
pickle_out.close()

#score_test =[]
#models = []

#k = np.range(1,30)

#for i in range(k):
#    classifer = KNeighborsClassifier(neighbors=1)
#    classifer.fit(X_train, y_train)
#    score_test.append()
#    models.append(classifer)

#pikcle.dump(model, open)