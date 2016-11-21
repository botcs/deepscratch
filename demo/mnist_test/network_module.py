import sys
import layer_module as lm
import conv_module as cm
import numpy as np
import dill
from utilities import StatusBar, ensure_dir


def load(load):
    return dill.load(open(load, 'rb'))


class network(object):
    '''Layer manager object

    despite layers can be handled as separate instances, adding new
    components, registering them, and training becomes inefficient for
    deep-network architects

    '''

    '''Network modification definitions'''
    def pop_top_layer(self):
        assert len(self.layerlist) > 2, \
            'No hidden layers to be popped'
        popped = self.top
        # using remove, pop would remove the OUTPUT layer which is forbidden
        self.layerlist.remove(self.top)
        self.top = self.top.prev
        self.output.new_last_layer(self.top)
        return popped

    def register_new_layer(self, l):
        self.top.next = l
        l.prev = self.top
        self.top = l
        self.output.new_last_layer(l)
        self.layerlist.insert(-1, l)
        return l

    def add_conv(self, num_of_ker, kernel_shape,  **kwargs):
        self.register_new_layer(
            cm.Conv(num_of_ker, kernel_shape, prev=self.top, **kwargs))
        return self

    def add_maxpool(self, **kwargs):
        self.register_new_layer(
            cm.max_pool(prev=self.top, **kwargs))
        return self

    def add_wta(self, k, **kwargs):
        self.register_new_layer(lm.wta(k, prev=self.top, **kwargs))
        return self        

    def add_dropcon(self, p, **kwargs):
        self.register_new_layer(lm.dropcon(p, prev=self.top, **kwargs))
        return self

    def add_full(self, width, **kwargs):
        self.register_new_layer(
            lm.fully_connected(width=width, prev=self.top, **kwargs))
        return self

    def add_dropout(self, p, **kwargs):
        self.register_new_layer(lm.dropout(p, prev=self.top, **kwargs))
        return self

    def add_activation(self, type, **kwargs):
        self.register_new_layer(
            lm.activation(type, prev=self.top, **kwargs))
        return self

    def add_shaper(self, shape, **kwargs):
        self.register_new_layer(
            lm.shaper(shape, prev=self.top, **kwargs))
        return self

    def save_state(self, file_name):
        ensure_dir(file_name)
        if file_name:
            dill.dump(self, open(file_name, 'wb'))
        else:
            dill.dump(self, open(str(id(self)) + '.net', 'wb'))

    def __str__(self):
        res = 'Network ID: ' + str(id(self))
        res += '\nNetwork layout:\n'
        res += '-' * 30 + '\n'
        '--------------------------------'
        res += '\tINPUT  {}'.format(self.input.shape)
        for i, l in enumerate(self.layerlist[1:], start=1):
            res += 2 * ('\n\t   |') + '\n\t  |{}|'.format(i) + '\n  '
            res += l.__str__()

        res += '\n' + '-' * 30
        return res

    def __init__(self, in_shape, criterion, **kwargs):
        self.input = lm.input(in_shape)
        self.top = self.input
        self.output = lm.output(type=criterion, prev=self.top)
        self.layerlist = [self.input, self.output]

    def __getitem__(self, index):
        return self.layerlist[index]

    def __repr__(self):
        return self.__str__()

    def get_output(self, input):
        return self.output.get_output(input)

    def perc_eval(self, test_set):
        T = self.test_eval(test_set)
        return T * 100.0 / len(test_set)

    def test_eval(self, test_set):
        return np.count_nonzero(self.get_output(test_set[0]).argmax(axis=1) ==
                                test_set[1].argmax(axis=1))

    'NETWORK TRAINING METHODS'
    def SGD(self, train_policy, training_set,
            batch, rate, L2=False, L1=False, L05=False, reg=0,
            validation_set=None, epoch_call_back=None, **kwargs):

        for l in self.layerlist:
            'Set the training method for layers where it is implemented'
            assert L05 + L1 + L2 < 2, 'Regularisation cannot be mixed'
            try:
                if L2:
                    l.train = l.L2train
                elif L1:
                    l.train = l.L1train
                elif L05:
                    l.train = l.L05train
                else:
                    l.train = l.SGDtrain
            except AttributeError:
                continue

        'For eliminating native lists, and tuples'
        input_set = np.array(training_set[0])
        target_set = np.array(training_set[1])
        
        assert len(input_set) == len(target_set),\
            'input and training set is not equal in size'
        while train_policy(training_set, validation_set, **kwargs):
            num_of_batches = len(input_set)/batch
            for b in xrange(num_of_batches):
                ##for longer training some data should be useful
                'FORWARD'
                test = np.sum(self.get_output(input_set[b::num_of_batches]))
                assert not (np.isnan(test) or np.isinf(test)),\
                    "NaN found in output during train- shutting down..."
                print('\r   batch: {} of {}'.format(
                      b+1, num_of_batches)),
                sys.stdout.flush()

                'BACKWARD'
                self.input.backprop(target_set[b::num_of_batches])

                'PARAMETER GRADIENT ACCUMULATION'
                for l in self.layerlist:
                    l.train(rate=rate, reg=reg)

            if epoch_call_back:
                'Some logging function is called here'
                epoch_call_back()

    'NETWORK TRAINING POLICIES'
    def fix_epoch(self, training_set, validation_set, **kwargs):
        try:
            self.last_epoch += 1
            return self.last_epoch <= kwargs['epoch']
        except AttributeError:
            self.last_epoch = 1
            return True

    def fix_hit_rate(self, training_set, validation_set, **kwargs):
        return self.perc_eval(validation_set) < kwargs['valid']

    def stop_when_overfit(self, training_set, validation_set, **kwargs):
        try:
            prev_hit = self.last_hit
            self.last_hit = self.test_eval(validation_set)
            return prev_hit < self.last_hit
        except AttributeError:
            self.last_hit = self.test_eval(validation_set)

    'VISUALISATION'
    def max_act(self, layer_ind, activation_set, top=1):
        l = self[layer_ind]
        res = l.get_output(activation_set)
        return res.argsort(axis=0)[-top:]

    def get_one_hot(self, layer_ind, biased=False):
        '''get one-hot matrixes for each neuron in layer to prop back'''
        l = self[layer_ind]
        # number of neurons
        nn = np.prod(l.shape)
        # each neuron should have its own one-hot matrix so:
        # nn^2 zeros should do
        if biased:
            oh = np.ones(nn ** 2) * -1
        else:
            oh = np.zeros(nn ** 2)
        # change 1 to simulate the ideal activation for that neuron
        # move this 1's index by nn + 1 to do so
        
        oh[::nn + 1] = 1

        # return the well shaped form of one-hots
        return oh.reshape(l.shape + l.shape)

    def backprop_one_hot(self, layer_ind, top=1, biased=False):
        'Should be called after forwarding each neurons most intense input'
        # broadcast stands for multiple top activation for each neuron

        l = self[layer_ind]
        oh = self.get_one_hot(layer_ind, biased)
        oh = np.broadcast_to(oh, ((top,) + oh.shape)).reshape(-1, *l.shape)
        store = l.get_delta
        l.get_delta = lambda(x): oh
        res = self.input.backprop(None)
        l.get_delta = store
        return res

    def grad_ascent(self, layer_ind, activation_set, top=9, epoch=5, rate=0.1, biased=False):
        """get current layer's each neuron's strongest corresponding inputs'
        index in activation set

        Example:
        if layer's output is 10*24*24 for a sample, then the returned
        indices will be shaped 9*10*24*24, where the [:,3,10,10]
        reveals the index of the top 9 image (from the activation_set)
        that causes the maximum activation at output's 3rd feature
        map's [10:10] located neuron

        """

        ind = self.max_act(layer_ind, activation_set, top)
        """retrieve those images from the activation_set, and forward them
        again

        Example:
        get (9, 3, 24, 24, 1, 28, 28) shaped image set, where
        (9, 3, 24, 24, ...) is described in the previous comment,
        and for each entry there are (..., 1, 28, 28) one channeled,
        28x28 image"""
        input =  activation_set[ind].reshape(-1, *self.input.shape)
        
        """for correct batch inference it should be reshaped to (-1, 1, 28, 28)
        where '-1' stands for implicitly 9*3*24*24 (all that remains)
        """
        import sys
        
        l = self[layer_ind]
        for e in xrange(epoch):
            print '\r GA: ', e + 1, '    ', 
            sys.stdout.flush()
            l.get_output(input)
            delta = self.backprop_one_hot(layer_ind, top, biased).reshape(input.shape)
            
        
            input += rate * delta
            # normalize input
            # input -= input.min(axis=0)
            # input /= input.max(axis=0)
        print ' --- Done!'
        return input
