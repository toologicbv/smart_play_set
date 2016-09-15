from pymongo import MongoClient
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


def plot_document(document):
  window = []
  for i in range(0,60):
    window.extend(document["data"][str(i)])

  y = np.array(window).astype(np.float).astype(np.float)

  x = np.arange(0, len(y))
      
  ax = plt.subplot(2,3,1)
  ax.set_title("ax")
  ax.plot(x, y[:,0])

  ay = plt.subplot(2,3,2)
  ay.set_title("ay")
  ay.plot(x, y[:,1])

  az = plt.subplot(2,3,3)
  az.set_title("az")
  az.plot(x, y[:,2])

  gx = plt.subplot(2,3,4)
  gx.set_title("gx")
  gx.plot(x, y[:,3])

  gy = plt.subplot(2,3,5)
  gy.set_title("gy")
  gy.plot(x, y[:,4])

  gz = plt.subplot(2,3,6)
  gz.set_title("gz")
  gz.plot(x, y[:,5])


client = MongoClient()
db = client.data
collection = db.arduino

documents = collection.find(); # for debug purpose only look at the first document

# plot some documents
plot_document(documents[17])
plot_document(documents[19])
plot_document(documents[0])
plot_document(documents[2])
plot_document(documents[7])

plt.show()
