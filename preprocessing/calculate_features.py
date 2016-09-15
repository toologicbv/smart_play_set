from __future__ import print_function

from pymongo import MongoClient
import numpy as np
import scipy.stats as sc
import matplotlib as plt
from sklearn import svm, naive_bayes, neighbors, tree, metrics, cross_validation

# this script shows how features can be extracted from the raw data collected in the database
# and how these features can be used to train a classifier
#
# the script requires:
# - a running instance of mongod (MongoDB server)
# 
# The data should be stored in a colletion called 'arduino'
#
# the features can either be stored in a new data base of preprocessed data or be used for classification.

client = MongoClient()
db = client.data
print("Current collections: ", db.collection_names(include_system_collections=False))
collection = db.arduino

for document in collection.find():
    print(document)


# ***** Calculating features *****

feature_vectors = []
labels = []

for document in collection.find(): #include a search object to select a specific subset of the data
#document = collection.find_one(); # for debug purpose only look at the first document

    for i in range(1, 60):
        maxf = 0.
        minf = 0.
        mean = 0.
        std = 0.
        median = 0.
        dc, E, H = 0., 0., 0.
        try:
          window = np.array(document["data"][str(i - 1)] + document["data"][str(i)]).astype(np.float) # 2s window, with overlap

          maxf = np.amax(window, 0) # maximum value for each of the features over the 2s window
          minf = np.amin(window, 0) # minimum value for each of the features
          mean = np.mean(window, 0)
          std = np.std(window, 0) # standard deviation
          median = np.median(window, 0)

          fd = np.fft.fft(window,len(window), 0) # frequency domain

          dc = np.real(fd[0]) # dc components

          ps = np.abs(fd)**2 # power spectrum

          E = np.sum(ps, 0) / len(window) # Energy (following Bao and Intille, is this correct?)

          H = sc.entropy(ps/np.sum(ps, 0)) # Spectral Entropy - should probably remove first coeff.

        except:
          print("warning: incomplete document, skipped") # data recording may stop before a document is completely filled

        feature_vectors.append(np.concatenate((maxf, minf, mean, std, median, dc, E, H)))
        labels.append(document["label"])

# ***** Classification *****

#classifier = svm.SVC(gamma=0.001, C=100.)
classifier = naive_bayes.GaussianNB()
#classifier = neighbors.KNeighborsClassifier()
#classifier = tree.DecisionTreeClassifier()

print(len(labels), "samples")

result = cross_validation.cross_val_predict(classifier, feature_vectors, labels)

print("accuracy:", metrics.accuracy_score(labels, result))
print(metrics.classification_report(labels, result))
