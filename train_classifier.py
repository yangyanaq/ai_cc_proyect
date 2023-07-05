import pickle
from yellowbrick.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split ,LearningCurveDisplay)

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier(n_estimators=3,max_depth=8)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print('Precision: %.3f' % precision_score(y_test, y_predict,average='micro'))
print('Recall: %.3f' % recall_score(y_test, y_predict,average='micro'))
print('F1: %.3f' % f1_score(y_test, y_predict,average='micro'))
print('Accuracy: %.3f' % accuracy_score(y_test, y_predict))


print('{}% de clasificacion fue correcta'.format(score * 100))
print(learning_curve(model, x_test, y_test, cv=10, scoring='accuracy'))



#f = open('model.p', 'wb')
#pickle.dump({'model': model}, f)
#f.close()
