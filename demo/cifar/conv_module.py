from scipy.signal import convolve2d
import numpy as np
import layer_module as lm

np.set_printoptions(precision=2, edgeitems=2, threshold=5)


class Conv(lm.AbstractLayer):

    def __init__(self, num_of_featmap, kernel_shape, **kwargs):
        lm.AbstractLayer.__init__(self, **kwargs)
        self.type = 'convolution'
        self.kernel_shape = kernel_shape
        self.nof = num_of_featmap

        if self.prev:
            self.shape = (self.nof,)
            self.shape += tuple(np.add(np.subtract(
                self.prev.shape[1:], self.kernel_shape), 2*(1,)))
            '''First parameter is the number of corresponding feature maps
            The remaining is the shape of the feature maps

            The output of 2D convolution on an input with
            shape [NxM]
            kernel [kxl]
            results in [(N-k+1) x (M-l+1)]'''

            'For fully connected next layer'
            self.width = np.prod(self.shape)
            self.bias = np.random.randn(self.nof)
            
            "prev layer's shape[0] is the number of output channels/feature maps"
            if kwargs.get('gaussian'):
                self.kernels = np.random.randn(
                    self.prev.shape[0], self.nof, *self.kernel_shape)
            elif kwargs.get('identity'):
                self.kernels = np.zeros((
                    self.prev.shape[0], self.nof) + self.kernel_shape, 
		            dtype=float)
            else:
                self.kernels = np.random.rand(
                    self.prev.shape[0], self.nof, *self.kernel_shape)

            if kwargs.get('sharp'):
                'Sharpening the deviation of initial values - regularization'
                self.kernels /= self.width

    def get_local_output(self, input):
        assert type(input) == np.ndarray

        'input will be required for training'
        if len(input.shape) == 3:
            'if input is a single sample, extend it to a 1 sized batch'
            self.input = np.expand_dims(input, 0)
        else:
            self.input = input

        '''Each channel has its corresponding kernel in each feature map
        feature map activation is evaluated by summing their activations
        for each sample in input batch'''
        return np.sum(
            [[[convolve2d(channel, kernel[::-1, ::-1], 'valid') #+ bias
               for kernel in kernel_set]
              for channel, kernel_set, bias in zip(sample, self.kernels, self.bias)]
             for sample in self.input], axis=1)

    def backprop_delta(self, delta):
        '''Each feature map is the result of all previous layer maps,
        therefore the same gradient has to be spread for each'''
        # THE SCIPY convolve2d IS REVERSING THE KERNEL AUTOMATICALLY
        return np.sum([
            [[convolve2d(k, d)
              for d, k in zip(sample_delta, kernel_set)]
             for kernel_set in self.kernels]
            for sample_delta in delta], axis=2)
        # saturating delta over 5 to prevent exploding gradient

    def get_param_grad(self):
        # THE SCIPY convolve2d IS REVERSING THE KERNEL AUTOMATICALLY
        return (np.array(
            # KERNEL GRAD
            [[[convolve2d(channel, d[::-1, ::-1], 'valid')
               for d in sample_delta]
              for channel in sample_input]
             for sample_input, sample_delta in zip(self.input, self.delta)]),
            # BIAS GRAD
            np.sum(self.delta, axis=(1, 2, 3)))

    def SGDtrain(self, rate, **kwargs):
        k_update, b_update = self.get_param_grad()
        self.kernels -= rate * k_update.mean(axis=0)
        self.bias -= rate * b_update.mean(axis=0)

#    def L2train(self, rate, reg):
#        k_update, b_update = self.get_param_grad()
#        self.kernels -= rate * k_update +\
#                        self.kernels * (rate * reg) / len(self.delta)

    def __str__(self):
        res = lm.AbstractLayer.__str__(self)
        res += '   ->   kernels: {}'.format(self.kernels.shape)
        return res


class max_pool(lm.AbstractLayer):

    def __init__(self, pool_shape=(2, 2), shape=None, **kwargs):
        lm.AbstractLayer.__init__(self, shape=shape, type='max pool', **kwargs)
        assert (shape is None) ^ (pool_shape is None),\
            "'pool_shape=' XOR 'shape=' must be defined"

        if self.prev:
            if shape:
                sp = np.divide(self.prev.shape, shape)
                '''First dimension is the number of feature maps in the previous
                   layer'''
                self.pool_shape = tuple(sp[1:])
            else:
                self.pool_shape = pool_shape
                self.shape = tuple(np.divide(self.prev.shape, (1,)+pool_shape))
                self.width = np.prod(self.shape)

    def get_local_output(self, input):

        if len(input.shape) == 3:
            'if input is a single sample, extend it to a 1 sized batch'
            x = np.expand_dims(input, 0)
        else:
            x = input

        batch, N, h, w = x.shape
        n, m = self.pool_shape
        'Reshape for pooling'
        x_col = input.reshape(batch, N, h/n, n, w/m, m)\
                     .transpose(0, 1, 2, 4, 3, 5)\
                     .reshape(batch, N, h/n, w/m, n*m)
        self.switch = np.argmax(x_col, axis=4)
        'Keep record of which neurons were chosen in the pool by their index'
        i = np.indices(self.switch.shape)
        return x_col[i[0], i[1], i[2], i[3], self.switch]

    def backprop_delta(self, delta):
        batch, N, h, w = self.prev.output.shape
        n, m = self.pool_shape
        res = np.zeros((batch, N, h/n, w/m, n*m))
        i = np.indices(self.switch.shape)
        res[i[0], i[1], i[2], i[3], self.switch] = delta

        return res.reshape(batch, N, h/n, w/m, n, m)\
                  .transpose(0, 1, 2, 4, 3, 5)\
                  .reshape(batch, N, h, w)

    def __str__(self):
        res = lm.AbstractLayer.__str__(self)
        res += '   ->   pool shape: {}'.format(self.pool_shape)
        return res
