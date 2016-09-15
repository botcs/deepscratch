import sys
import numpy as np
import network_module as nm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '-L1', type=float
)
parser.add_argument(
    '-L2', type=float
)
parser.add_argument(
    '-L05', type=float
)
parser.add_argument(
    'width', nargs='+', type=int
)
args = parser.parse_args(['50', '-L2', '0.01'])


def print_csv(filename, data):
    with open(filename, 'wb') as out:
        for t in data:
            out.write('{}\t{}\n'.format(*t))


# layer_params = sys.argv[1:]

# netname = 'mnist-fc-784-'
# for width in layer_params:
#     netname += '{}-'.format(width)
# netname += '10'

L1 = False
L2 = False
L05 = False
netname = ''


if args.L1:
    L1 = True
    netname += 'L1'
    reg = args.L1
elif args.L2:
    L2 = True
    netname += 'L2'
    reg = args.L2
elif args.L05:
    L05 = True
    netname += 'L05'
    reg = args.L05
else:
    netname += 'noreg'

if L1 or L2 or L05:
    netname += '-' + str(reg)[2:] + '-'


for width in args.width:
    netname += '-{}'.format(width)


def loadmnist():
    import cPickle, gzip

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

for width in args.width:
    nn.add_full(width)
    nn.add_activation('tanh')

# nn.add_activation('logistic')

nn.add_full(10)
#########################
print nn

result = []


def print_test():
    print nn.last_epoch, ' ', nn.test_eval(test)
    result.append((nn.test_eval(train), nn.test_eval(test)))


print 'Training network', netname
nn.SGD(train_policy=nn.fix_epoch,
       training_set=train,
       L2=L2, L1=L1, L05=L05, reg=reg,
       batch=16, rate=0.05, epoch_call_back=print_test, epoch=10)

print 'Saving results to {}.res'.format(netname)
print_csv('./results/{}.res'.format(netname), result)

print 'Saving network snapshot to {}.net'.format(netname)
nn.save_state('./nets/' + netname + '.net')
