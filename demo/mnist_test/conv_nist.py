import numpy as np
import matplotlib.pyplot as plt

'must be in the working directory, or in the python path'
import network_module as nm

netname = 'newtest---25conv3-10conv3-10conv-5'

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
trunc=50000
train = (train[0][0:trunc], train[1][0:trunc])
valid = (valid[0][0:trunc/10], valid[1][0:trunc/10])
print 'constructing network'
#########################
# NETWORK DEFINITION
nn = nm.network(in_shape=train[0][0].shape, criterion='softmax')
nn.add_conv(25, (3, 3), sharp=True)
nn.add_activation('relu')
nn.add_conv(10, (3, 3), sharp=True)
nn.add_activation('relu')
nn.add_conv(10, (5, 5), sharp=True)
nn.add_activation('relu')
nn.add_shaper(np.prod(nn[-1].shape))
nn.add_activation('tanh')
nn.add_full(10, sharp=True)
#########################
print nn

result = []


def print_test(loss_list):
    global result
    result.append(np.mean(loss_list))
    print ' --- Epoch: ', nn.last_epoch,\
    '  Mean loss: ', np.mean(loss_list)
    
def print_loss(loss):
    global result
    result += loss
    print loss

print 'Working with network:', netname
def train_net():
  print 'Training network:', netname
  nn.SGD(train_policy=nn.fix_epoch,
         training_set=train,
         batch=20, rate=0.01, 
         epoch_call_back=print_test, 
         epoch=25)
         
  nn.save_state('./nets/{}.net'.format(netname))       
         
         
def imshow(im, cmap='Greys_r', interpol='None'):
    import matplotlib.pyplot as plt
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
    __main__()
    
