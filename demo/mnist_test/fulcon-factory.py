import sys
import numpy as np
import network_module as nm


def print_csv(filename, data):
    with open(filename, 'wb') as out:
        for t in data:
            out.write('{}\t{}\n'.format(*t))


layer_params = sys.argv[1:]

netname = 'mnist-fc-784-'
for width in layer_params:
    netname += '{}-'.format(width)
netname += '10'


def loadmnist():
    import cPickle, gzip, numpy

    # Load the dataset
    f = gzip.open('./MNIST_data/mnist.pkl.gz', 'rb')
    sets = cPickle.load(f)
    f.close()
    res = []
    for set in sets:
        set[0].shape = (-1, 1, 28, 28)

        label = set[1]
        onehot = np.zeros((label.size, label.max() + 1))
        onehot[np.arange(label.size), label] = 1
        res.append((set[0].reshape(-1, 1, 28, 28), onehot))
    return res


print 'Loading MNIST images...'
train, valid, test = loadmnist()


print 'Constructing network...'
#########################
# NETWORK DEFINITION
nn = nm.network(in_shape=train[0][0].shape, criterion='softmax')
nn.add_shaper(np.prod(nn[-1].shape))

for width in layer_params:
    nn.add_full(int(width))
    nn.add_activation('relu')

nn.add_full(10)
#########################
print nn

result = []


def print_test():
    print nn.last_epoch, ' ', nn.test_eval(test)
    result.append((nn.test_eval(train), nn.test_eval(test)))


print 'Training network', netname
nn.SGD(train_policy=nn.fix_epoch, training_set=train,
       batch=16, rate=0.05, epoch_call_back=print_test, epoch=4)

print 'Saving results to {}.res'.format(netname)
print_csv('./results/{}.res'.format(netname), result)

print 'Saving network snapshot to {}.net'.format(netname)
nn.save_state('./nets/' + netname + '.net')
