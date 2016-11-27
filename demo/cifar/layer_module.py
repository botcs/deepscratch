import numpy as np
import utilities as util
import warnings


class AbstractLayer:

    def __init__(self, **kwargs):
        'get returns None instead of raising KeyError'
        self.type = kwargs.get('type')
        self.prev = kwargs.get('prev')
        self.next = kwargs.get('next')

        self.width = kwargs.get('width')
        self.shape = kwargs.get('shape')
        if self.prev is None:
            util.warning('No previous layer defined', self)

        'Shape states for the default output of the layer'
        if self.shape and not self.width:
            self.width = np.prod(self.shape)
        elif self.width:
            self.shape = (self.width)

        'Shape must be always a tuple... numpy syntax will blow up else'
        if type(self.shape) is not tuple:
            self.shape = (self.shape,)

    def get_propagated_input(self, input):
        '''Recursive definition, a layer can only produce real output from the
        previous layer.

        The only case when the input is directly calculated through
        one layer, is when there is no previous layer

        '''
        if self.prev:
            input = self.prev.get_output(input)

        return input

    def get_local_output(self, input):
        pass

    def get_param_grad(self):
        pass

    def acc_grad(self):
        pass

    def get_output(self, input):
        prop_in = self.get_propagated_input(input)
        self.output = self.get_local_output(prop_in)
        return self.output

    def get_delta(self, target):
        if self.next:
            # clip to mitigate exploding gradients
            self.delta = np.clip(self.next.backprop(target), -5, 5)
            if np.count_nonzero(self.delta) == 0:
                print 'WARNING! delta has no nonzero element!'
            return self.delta

        return target

    def backprop_delta(self, delta):
        pass

    def backprop(self, target):
        return self.backprop_delta(self.get_delta(target))

    def train(self, rate, **kwargs):
        pass

    def __str__(self):
        res = '{}  {}'.format(self.type, self.shape)
        return res


class fully_connected(AbstractLayer):

    def __init__(self, **kwargs):
        AbstractLayer.__init__(self, type='fully connected', **kwargs)
        '''ROW REPRESENTS OUTPUT NEURON'''
        assert self.width is not None, "'shape=' or 'width=' must be defined"
        if self.prev:
            if kwargs.get('gaussian'):
                self.weights = np.random.randn(self.width, self.prev.width)
                self.bias = np.random.randn(self.width)
            else:
                self.weights = np.random.rand(self.width, self.prev.width)    
                self.bias = np.random.rand(self.width)
            
            if kwargs.get('sharp'):
                'Sharpening the deviation of initial values - regularization'
                self.weights /= self.width

    def perturb(self, delta):
        'strictly experimental'
        self.weights += (np.random.randn(*self.weights.shape)) * delta

    def get_local_output(self, input):
        assert type(input) == np.ndarray, 'Only numpy array can be used'

        'input will be required for training'
        if len(input.shape) == 1:
            'if input is a single sample, extend it to a 1 sized batch'
            self.input = np.expand_dims(input, 0)
        else:
            self.input = input

        return np.dot(self.input, self.weights.T) + self.bias

    def backprop_delta(self, delta):
        assert self.next, 'Missing next layer, delta cannot be evaluated'
        return np.dot(delta, self.weights)

    def get_param_grad(self):
        'weight and bias gradient'
        return (np.mean([np.outer(i, d) for i, d in
                         zip(self.input, self.delta)], axis=0).T,
                np.mean(self.delta, axis=0))

    def L2train(self, rate, reg):
        w_grad, b_grad = self.get_param_grad()
        self.weights -= rate * (reg * self.weights + w_grad)
        self.bias -= rate * b_grad

    def L1train(self, rate, reg):
        w_grad, b_grad = self.get_param_grad()
        self.weights -= rate * (reg * np.sign(self.weights) + w_grad)
        self.bias -= rate * b_grad

    def L05train(self, rate, reg):
        w_grad, b_grad = self.get_param_grad()
        self.weights -= rate * (reg * self.weights /
                                np.power(np.abs(self.weights), 3/2.) + w_grad)
        self.bias -= rate * b_grad
        

    def SGDtrain(self, rate, reg):
        '''GRADIENT DESCENT TRAINING'''
        w_grad, b_grad = self.get_param_grad()
        self.weights -= rate * w_grad
        self.bias -= rate * b_grad

    def __str__(self):
        res = AbstractLayer.__str__(self)
        res += '   ->   weights + bias: {} + {}'.format(
            self.weights.shape, self.bias.shape)
        return res


class activation(AbstractLayer):
    '''
    STRICTLY ONE-TO-ONE CONNECTION, WITH f ACTIVATION FUNCTIONS

    IN1 ------ [f] ------ OUT1
    IN2 ------ [f] ------ OUT2
    IN3 ------ [f] ------ OUT3
    IN4 ------ [f] ------ OUT4
    '''
    activation_functions = {
        'identity': lambda x: x,
        'relu': lambda x: x * (x > 0),
        'tanh': lambda x: np.tanh(x),
        'atan': lambda x: np.arctan(x),
        'logistic': lambda x: 1.0 / (1.0 + np.exp(-x)),
    }

    derivative_functions = {
        'identity': lambda x: np.ones_like(x),
        'relu': lambda x: 1.0 * (x > 0),
        'tanh': lambda x: 1.0 - np.square(np.tanh(x)),
        'atan': lambda x: 1.0 / (1.0 + x**2),
        'logistic': lambda x: activation.activation_functions['logistic'](x) *
            (1 - activation.activation_functions['logistic'](x))
    }

    def act(self, x):
        return activation.activation_functions[self.type](x)

    def der(self, x):
        return activation.derivative_functions[self.type](x)

    def __init__(self, type, **kwargs):
        if kwargs.get('prev'):
            prev_shape = kwargs.get('prev').shape
            AbstractLayer.__init__(self, shape=prev_shape, type=type, **kwargs)
        else:
            AbstractLayer.__init__(self, type=type, **kwargs)

    def __str__(self):
        return 'activation {}   ->   type: {}'.format(self.shape, self.type)

    def get_local_output(self, input):
        self.input = input.mean(axis=0)
        return self.act(input)

    def backprop_delta(self, delta):
        return self.der(self.input) * delta


class shaper(AbstractLayer):
    def __init__(self, shape, **kwargs):
        AbstractLayer.__init__(self, shape=shape, type='shaper', **kwargs)
        assert np.prod(self.prev.shape) == np.prod(shape),\
            'The shaper cannot modify the size of the output:{} vs. {}'\
            .format(np.prod(self.prev.shape), np.prod(shape))

    def get_local_output(self, input):
        """Only purpose of this layer is forced shaping of the input useful
        when processing a batch input, where a normal input shape
        would be (S1, S2, ...) and the batch size is B then the batch
        input's shape is (B, S1, S2, ...).

        The output shape will always be (B, S1', S2', ...)  where
        product of S' IS equal to the product of S, and the first
        dimension's size doesn't change.

        I.e. flattening for a fully connected layer the input
        should be flattened in all axes, but the first one.

        """
        return input.reshape(-1, *self.shape)

    def backprop_delta(self, delta):
        return delta.reshape(-1, *self.prev.shape)


class input(activation):

    def __init__(self, shape, **kwargs):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            activation.__init__(self, shape=shape,
                                type='identity', **kwargs)
                                
    def get_local_output(self, input):
        self.input = input
        if len(input.shape) == 3:
            return input[None, ...]
        return input

    def backprop_delta(self, delta):
        return delta.reshape(-1, *self.shape)


class output(activation):
    '''Special type of activation layer, that has TWO kind of output:

    FIRST is [N -> N] type, very similar to activation, but the
    activation function is not strictly one-to-one

    SECOND is [(N x N) -> 1] always, producing a
    Loss/Criterion/Cost/Error like result as a function of the FIRST
    output, and the Target

    '''

    def __init__(self, **kwargs):
        activation.__init__(self, **kwargs)

    def __str__(self):
        res = '\tOUTPUT  {}'.format(self.shape)
        res += '   ->   '
        res += 'CRITERION  ({})'.format(self.type, self.shape, 1)
        return res

    def new_last_layer(self, new_layer):
        self.prev = new_layer
        self.prev.next = self
        self.width = new_layer.width
        self.shape = new_layer.shape

    crit = {
        'MSE': lambda (prediction, target):
            0.5 * np.sum((prediction - target)**2),
        'softmax': lambda (prediction, target):
            np.sum(-target * np.log(prediction), axis=1)
    }

    activation = {
        'MSE': lambda x: x,
        'softmax': lambda x: np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)), axis=1)[:, np.newaxis]
    }

    derivative = {
        'MSE': lambda (prediction, target): prediction - target,
        'softmax': lambda (prediction, target): prediction - target
    }

    '''
    for later expansion of these functions
    https://github.com/torch/nn/blob/master/doc/criterion.md
    '''

    def get_local_output(self, input):
        return output.activation[self.type](input)

    def get_crit(self, target):
        assert self.output.shape == target.shape, \
            'last inference SHAPE {} does not match target SHAPE {}'.\
            format(self.output.shape, target.shape)
        'double paren for lambda wrap'
        return output.crit[self.type]((self.output, target))

    def backprop_delta(self, delta):
        '''The delta of the output layer wouldn't be used for training
        so the function returns directly the delta of the previous layer
        '''
        self.prev_delta = output.derivative[self.type]((self.output, delta)).mean(axis=0)[None]
        return self.prev_delta

class batchnorm(activation):
    def __init__(self, mean=0.5, std=1.0, **kwargs):
        activation.__init__(self, type='batch norm', **kwargs)
        self.mean = mean
        self.std = std
        
    def __str__(self):
        return 'batch norm {}   ->   mean = {}, std = {}'.\
            format(self.shape, self.mean, self.std)
        
    def get_local_output(self, input):
        self.input = input.mean(axis=0)
        self.batch_std = input.std()
        res = input / self.batch_std / self.std
        return res - res.mean() + self.mean
        
    def backprop_delta(self, delta):
        return delta * self.input / self.batch_std / self.std

class wta(activation):

    'K-WTA implementation - for flat batch input: (batch_size, prev_layer_width)'
    def __init__(self, k, **kwargs):
        activation.__init__(self, type='wta', **kwargs)
        self.k = k


    def __str__(self):
        return 'Winner Takes All   ->   k = {}'.format(self.k)


    def get_local_output(self, input):
        ind = input.argpartition(-self.k, axis=1)[:, -self.k:]
        i = np.arange(input.shape[0])[:,None]
    
        self.mask = np.zeros_like(input, dtype=bool)
        self.mask[i, ind] = True
        
        # prevent saturation when the layer is removed
        return input * self.mask * (input.shape[1]/self.k)

    def backprop_delta(self, delta):
        return delta * self.mask 
    

class dropout(activation):

    def __init__(self, p, type='dropout', **kwargs):
        activation.__init__(self, **kwargs)
        self.p = p

    def get_local_output(self, input):
        self.curr_drop = np.random.rand(*input.shape) > self.p
        # divide by p for keeping the next layer unsaturated
        return input * self.curr_drop / self.p

    def backprop_delta(self, delta):
        return self.curr_drop * delta


class dropcon(fully_connected):

    def __init__(self, p, type='dropcon', **kwargs):
        fully_connected.__init__(self, **kwargs)
        self.p = p

    def get_local_output(self, input):
        self.curr_drop = np.random.rand(*self.weights.shape) > self.p
        self.input = input
        return np.dot((self.curr_drop * self.weights), input)

    def backprop_delta(self, delta):
        return np.dot(delta, self.curr_drop * self.weights)
