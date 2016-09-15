## -------2 HIDDEN LAYER----------

# -----TESTING WITH THE SAME NUMBER OF NEURONS (CORRESPONDING TO 1HL NETS - NOT 1:1)
# IN CASE IF CORRESPONDING NET IS MISSING H1 NET IS TRAINED IN THIS SCRIPT
nohup ipython fulcon-factory.py 20 > ./logs/mnist-20.log 2>&1&
nohup ipython fulcon-factory.py 15 5 > ./logs/mnist-15-5.log 2>&1&
nohup ipython fulcon-factory.py 10 10 > ./logs/mnist-5-5.log 2>&1&
nohup ipython fulcon-factory.py 5 15 > ./logs/mnist-5-15.log 2>&1&

nohup ipython fulcon-factory.py 40 10 > ./logs/mnist-40-10.log 2>&1&
nohup ipython fulcon-factory.py 30 20 > ./logs/mnist-30-20.log 2>&1&
nohup ipython fulcon-factory.py 25 25 > ./logs/mnist-25-25.log 2>&1&
nohup ipython fulcon-factory.py 20 30 > ./logs/mnist-20-30.log 2>&1&
nohup ipython fulcon-factory.py 10 40 > ./logs/mnist-10-40.log 2>&1&

nohup ipython fulcon-factory.py 150 50 > ./logs/mnist-150-50.log 2>&1&
nohup ipython fulcon-factory.py 125 75 > ./logs/mnist-125-75.log 2>&1&
nohup ipython fulcon-factory.py 100 100 > ./logs/mnist-100-100.log 2>&1&
nohup ipython fulcon-factory.py 75 125 > ./logs/mnist-75-125.log 2>&1&
nohup ipython fulcon-factory.py 50 150 > ./logs/mnist-50-150.log 2>&1&


nohup ipython fulcon-factory.py 400 100 > ./logs/mnist-400-100.log 2>&1&
nohup ipython fulcon-factory.py 300 200 > ./logs/mnist-300-200.log 2>&1&
nohup ipython fulcon-factory.py 250 250 > ./logs/mnist-250-250.log 2>&1&
nohup ipython fulcon-factory.py 200 300 > ./logs/mnist-200-300.log 2>&1&
nohup ipython fulcon-factory.py 100 400 > ./logs/mnist-100-400.log 2>&1&

# IN CASE ANY AUTOENCODER PATTER WOULD SHOW UP...
nohup ipython fulcon-factory.py 784 784 > ./logs/mnist-784.log 2>&1&

# WIDER-THAN-INPUT NETS
nohup ipython fulcon-factory.py 1000 100 > ./logs/mnist-1000-100.log 2>&1&
nohup ipython fulcon-factory.py 1000 200 > ./logs/mnist-1000-200.log 2>&1&
nohup ipython fulcon-factory.py 1000 500 > ./logs/mnist-1000-500.log 2>&1&
nohup ipython fulcon-factory.py 1000 784 > ./logs/mnist-1000-784.log 2>&1&
nohup ipython fulcon-factory.py 1000 1000 > ./logs/mnist-1000-1000.log 2>&1&
nohup ipython fulcon-factory.py 2000 1000 > ./logs/mnist-2000-1000.log 2>&1&
nohup ipython fulcon-factory.py 1000 2000 > ./logs/mnist-1000-2000.log 2>&1&

