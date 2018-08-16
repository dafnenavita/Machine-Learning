
# coding: utf-8

# In[281]:


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc

text_label = pd.read_csv('yelp_labelled.txt', header=None, delimiter='\t')
train, test = train_test_split(text_label, train_size = 0.8)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train[0])
print(count_vect.get_feature_names())
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
clf = MultinomialNB(alpha=0.5)
clf.fit(X_train_tfidf, train[1])
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB(alpha=0.5)),
])
print(text_clf.fit(train[0], train[1]))

docs_test = test[0]
predicted = text_clf.predict(docs_test)
print("Accuracy: %0.2f" % (np.mean(predicted == test[1])))
print("Classification Report")
print(metrics.classification_report(test[1], predicted,
    target_names=['0', '1']))
print("Confusion Metrics")
print(metrics.confusion_matrix(test[1], predicted))

# calculate the fpr and tpr for all thresholds of the classification
probs = text_clf.predict_proba(test[0])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test[1], preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

