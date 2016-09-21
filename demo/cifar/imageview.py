def load(file):
  import cPickle, gzip
  # Load the dataset
  f = open(file, 'rb')
   
  res = cPickle.load(f)
  return res

import numpy as np
import matplotlib.pyplot as plt
import network_module as nm

def p(i, **kwargs):
  print(i.shape)
  plt.show(plt.imshow(i, **kwargs))
 

  
