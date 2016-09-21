## ---------- ConvNet SCRIPTS ------------
# -------FEATURE MAP TEST ---------
# nohup python netfactory.py c 3 10 > ./logs/conv/mnist-c3x10.log 2>&1&
# nohup python netfactory.py c 5 10 > ./logs/conv/mnist-c5x10.log 2>&1&
# nohup python netfactory.py c 10 10 > ./logs/conv/mnist-c10x10.log 2>&1&
# nohup python netfactory.py c 30 10 > ./logs/conv/mnist-c30x10.log 2>&1&
# nohup python netfactory.py c 50 10 > ./logs/conv/mnist-c50x10.log 2>&1&
# 
# nohup python netfactory.py c 3 5 > ./logs/conv/mnist-c3x10.log 2>&1&
# nohup python netfactory.py c 5 5 > ./logs/conv/mnist-c5x10.log 2>&1&
# nohup python netfactory.py c 10 5 > ./logs/conv/mnist-c10x10.log 2>&1&
# nohup python netfactory.py c 30 5 > ./logs/conv/mnist-c30x10.log 2>&1&
# nohup python netfactory.py c 50 5 > ./logs/conv/mnist-c50x10.log 2>&1&
# 
# # -------KERNEL SIZE TEST ---------
# nohup python netfactory.py c 20 3 > ./logs/conv/mnist-c20x3.log 2>&1&
# nohup python netfactory.py c 20 4 > ./logs/conv/mnist-c20x4.log 2>&1&
# nohup python netfactory.py c 20 5 > ./logs/conv/mnist-c20x5.log 2>&1&
# nohup python netfactory.py c 20 6 > ./logs/conv/mnist-c20x6.log 2>&1&
# nohup python netfactory.py c 20 7 > ./logs/conv/mnist-c20x7.log 2>&1&
# nohup python netfactory.py c 20 8 > ./logs/conv/mnist-c20x8.log 2>&1&
# nohup python netfactory.py c 20 9 > ./logs/conv/mnist-c20x9.log 2>&1&
# nohup python netfactory.py c 20 10 > ./logs/conv/mnist-c20x10.log 2>&1&
# nohup python netfactory.py c 20 11 > ./logs/conv/mnist-c20x11.log 2>&1&
 
# -------RECOGNITION LAYER TEST ------
nohup python netfactory.py c 15 5 f 10 > ./logs/conv/mnist-c15x5-f10.log 2>&1&
nohup python netfactory.py c 15 5 f 15 > ./logs/conv/mnist-c15x5-f15.log 2>&1&
nohup python netfactory.py c 15 5 f 20 > ./logs/conv/mnist-c15x5-f20.log 2>&1&
nohup python netfactory.py c 15 5 f 40 > ./logs/conv/mnist-c15x5-f40.log 2>&1&
nohup python netfactory.py c 15 5 f 50 > ./logs/conv/mnist-c15x5-f50.log 2>&1&
nohup python netfactory.py c 15 5 f 80 > ./logs/conv/mnist-c15x5-f80.log 2>&1&
nohup python netfactory.py c 15 5 f 100 > ./logs/conv/mnist-c15x5-f100.log 2>&1&
nohup python netfactory.py c 15 5 f 150 > ./logs/conv/mnist-c15x5-f150.log 2>&1&

nohup python netfactory.py c 10 10 f 15 > ./logs/conv/mnist-c10x10-f15.log 2>&1&
nohup python netfactory.py c 10 10 f 20 > ./logs/conv/mnist-c10x10-f20.log 2>&1&
nohup python netfactory.py c 10 10 f 40 > ./logs/conv/mnist-c10x10-f40.log 2>&1&
nohup python netfactory.py c 10 10 f 50 > ./logs/conv/mnist-c10x10-f50.log 2>&1&
nohup python netfactory.py c 10 10 f 80 > ./logs/conv/mnist-c10x10-f80.log 2>&1&
nohup python netfactory.py c 10 10 f 100 > ./logs/conv/mnist-c10x10-f100.log 2>&1&
nohup python netfactory.py c 10 10 f 150 > ./logs/conv/mnist-c10x10-f150.log 2>&1&



