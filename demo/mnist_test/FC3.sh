## -------3 HIDDEN LAYERS-----------
declare -a parameters=(
"125 100 75"
"100 100 100"
"75 100 125"

"350 200 50"
"200 200 200"
"50 200 350"

"600 200 100"
"300 300 300"
"100 200 600"

"1000 200 300"
"500 500 500"
"400 100 1000"

"100 784 784"
"784 100 784"
"784 784 100"
"784 784 784"

"1000 1000 1000"
)

for p in "${parameters[@]}"
do
   # START MULTI THREAD
   eval nohup ipython fulcon-factory.py "$p" > ./logs/mnist-"$p".log 2>&1&
done
