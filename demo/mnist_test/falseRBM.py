import numpy as np
import matplotlib.pyplot as plt

'must be in the working directory, or in the python path'
import network_module as nm

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

# truncating sets
trunc = 1000
train = (train[0][0:trunc], train[0][0:trunc])
valid = (valid[0][0:trunc/10], valid[0][0:trunc/10])
print 'constructing network'
#########################
# NETWORK DEFINITION
nn = nm.network(in_shape=train[0][0].shape, criterion='MSE')
nn.add_conv(3, (5, 5))
nn.add_maxpool()
nn.add_shaper(np.prod(nn[-1].shape))
nn.add_activation('relu')
nn.add_full(10)
nn.add_activation('relu')
nn.add_full(784)

nn.add_shaper(train[0][0].shape)
#########################
print nn

result = []


def print_test():
    # print nn.last_epoch, ' ', nn.test_eval(train)
    print(nn.last_epoch, nn.output.get_crit(train[0][0], train[0][0]))



def imshow(im, cmap='Greys_r', interpol='None'):
    import matplotlib.pyplot as plt
    if len(im.squeeze().shape) > 2:
        im = im.squeeze()
    if len(im.shape) == 3:
        for i, x in enumerate(im, 1):
            plt.subplot(1, len(im), i)
            plt.imshow(x.squeeze(), cmap=cmap, interpolation=interpol)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
    if len(im.shape) == 4:
        for irow, xrow in enumerate(im, 0):
            for icol, x in enumerate(xrow, 1):
                print '\r  ', len(im), len(xrow), irow * len(xrow) + icol
                plt.subplot(len(im), len(xrow), irow * len(xrow) + icol)
                plt.imshow(x.squeeze(), cmap=cmap, interpolation=interpol)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.show()
    return im.shape

# imshow(nn.get_output(test_data[0]))
# test = nn.get_output(test_data[0:3]) * 100
# im = zip(test_data[0:3], test, nn[1].backprop_delta(test))
# imshow(im[2])

def print_result():
    plt.show(plt.plot(result))

# print_csv('./test_runs/{}-rate005'.format(name), result)


def visualise_layer(lay_ind=4, top=9, iterations=10):
    test = nn.grad_ascent(lay_ind, train[0][0:trunc], top, iterations)\
             .reshape((top,) + nn[lay_ind].shape + (28, 28))
    return test


def max_act(lay_ind, top=9):
    return test[0][nn.max_act(lay_ind, test_data, top)].squeeze()
    

def __main__():
    loadmnist()

if __name__ == '__main__':
    main()
    

print 'Training network... '
nn.SGD(train_policy=nn.fix_epoch, training_set=train,
       batch=10, rate=0.005, epoch_call_back=print_test, epoch=100)

xy = np.array([train[0][0:10], nn.get_output(train[0][0:10]),
               train[0][10:20], nn.get_output(train[0][10:20])])
