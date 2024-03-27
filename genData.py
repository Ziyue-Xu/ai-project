from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


df = pd.read_csv('fuel.csv')
X = df['engine_displacement']
y = df['unrounded_city_mpg_ft1']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y,random_state=1)


dt =DecisionTreeClassifier(max_depth=2, random_state=1)
dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)
print(accuracy_score(y_test,y_pred))