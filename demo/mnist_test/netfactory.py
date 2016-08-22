import sys
import numpy as np
import network_module as nm

seq = sys.argv[1:]
# seq = ['c', '5', '10', 'f', '10']
i = 0


def print_csv(filename, data):
    with open(filename, 'wb') as out:
        for t in data:
            out.write('{}\t{}\n'.format(*t))


netname = 'mnist'
i = 0
while i < len(seq):
    if seq[i] == 'c':
        netname += '-c{}x{}'.format(seq[i+1],  seq[i+2])
        i += 3
    elif seq[i] == 'f':
        netname += '-f{}'.format(seq[i+1])
        i += 2
    else:
        print 'WARNING: Expected layer flag, got instead: \'' + seq[i] + '\''
        i += 1



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


print 'Constructing', netname, 'network...'
#########################
# NETWORK DEFINITION
nn = nm.network(in_shape=train[0][0].shape, criterion='softmax')
i = 0
while i < len(seq):
    if seq[i] == 'c':
        nn.add_conv(int(seq[i+1]), (int(seq[i+2]), int(seq[i+2])))
        i += 3
    elif seq[i] == 'f':
        nn.add_shaper(np.prod(nn[-1].shape))
        nn.add_full(int(seq[i+1]))
        i += 2
    else:
        print 'WARNING: Expected layer flag, got instead: \'' + seq[i] +'\''
        i += 1

    nn.add_activation('relu')


nn.add_shaper(np.prod(nn[-1].shape))
nn.add_full(10)
#########################
print nn

result = []


def print_test():
    train_score = nn.test_eval(train)
    test_score = nn.test_eval(test)
    print nn.last_epoch, train, test
    result.append(train_score, test_score)


print 'Training network', netname
nn.SGD(train_policy=nn.fix_epoch, training_set=train,
       batch=16, rate=0.05, epoch_call_back=print_test, epoch=4)

print 'Saving results to {}.res'.format(netname)
print_csv('./results/{}.res'.format(netname), result)

print 'Saving network snapshot to {}.net'.format(netname)
nn.save_state('./nets/' + netname + '.net')
