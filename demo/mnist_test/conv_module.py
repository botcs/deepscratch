import numpy as np
import layer_module as lm
from utilities import batch_im2col

np.set_printoptions(precision=2, edgeitems=2, threshold=10)

class Conv(lm.AbstractLayer):

    def __init__(self, num_of_featmap, kernel_shape, stride=(1, 1), **kwargs):
        lm.AbstractLayer.__init__(self, **kwargs)
        self.type = 'convolution'
        self.kernel_shape = kernel_shape
        self.depth = num_of_featmap
        self.stride = stride

        if self.prev:

            C, N, M = self.prev.shape
            n, m = self.kernel_shape
            S1, S2 = self.stride
            ver = (N-n)/S1 +1
            hor = (M-m)/S2 +1
            self.shape = (self.depth, ver, hor)    
            
            'For fully connected next layer'
            self.width = np.prod(self.shape)
            
            self.bias = np.random.randn(self.depth)
            "prev layer's shape[0] is the number of output channels/feature maps"

            if kwargs.get('gaussian'):
                self.kernels = np.random.randn((self.depth, C, n, m))
            elif kwargs.get('identity'):
                self.kernels = np.tile(np.eye(self.kernel_shape[0], dtype=float), 
                    (self.depth, C, 1, 1))
            else:
                self.kernels = np.random.rand(self.depth, C, n, m)

            self.bias = np.zeros(self.depth)            
            
            if kwargs.get('sharp'):
                'Sharpening the deviation of initial values - regularization'
                self.kernels /= self.depth

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
            
        '''im is a sample from a BATCH, with dimensions:
            C --- feature maps / color channels
            N --- common height of feature maps
            M --- common width of feature maps
            
            im.shape == (C, N, M)
        '''
        
        '''ker is the convolution kernel, has the following dimensions:
           C --- corresponding number of color channel / feature maps
           n --- common height of kernels
           m --- common width of kernels
           
           ker.shape == (C, n, m)
        '''
        
        '''stride is a tuple with S1, S2 values and determines
           in what manner the kernels will be applied to im
        '''
        
        '''Flatten 3D windows of kernels' perceptive field, with given stride
           stack them together into a 2D matrix, namely col
           
           W is determined by how many times the kernel could be applied
           to the image (with the given stride)
           
           vertical num of possibilities:
           ver = (N-n)/S1 +1
           
           horizontal num of possibilities: 
           hor = (M-m)/S2 +1
           
           W = ver * hor
           
           col.shape == (W, C*n*m)
        '''

        col = batch_im2col(self.input, self.kernel_shape, self.stride)\
            .swapaxes(1,2)
        
        batch = input.shape[0]                  
        output = np.inner(col, self.kernels.reshape(self.kernels.shape[0], -1)).\
            swapaxes(2,1).reshape((batch, ) + self.shape)
                    
        return output + self.bias[None, :, None, None]

    def backprop_delta(self, delta):
        '''Each feature map is the result of all previous layer maps,
        therefore the same gradient has to be spread for each'''
        
        '''Pad upper left corner for back-prop'''
        n, m = self.kernel_shape
        if n > m:
            padded = np.pad(delta, n-1, 'constant')[n-1:1-n, n-1:1-n, :, n-m:m-n]
        elif n < m:
            padded = np.pad(delta, m-1, 'constant')[m-1:1-m, m-1:1-m, m-n:n-m, :]
        else:
            padded = np.pad(delta, m-1, 'constant')[m-1:1-m, m-1:1-m]
        
        padded = padded[..., ::-1, ::-1]
        batch = delta.shape[0]
        
        col = batch_im2col(padded, self.kernel_shape, self.stride).swapaxes(1,2)
                                
        
        '''Now deltas of different feature map are the Channels
           and kernels of different feature map corresponding 
           to the same original Channel are the new kernel columns
           and original Channels are now the feature map
        '''
        
        swapker = self.kernels.swapaxes(0,1)
        output = np.inner(col, swapker.reshape(swapker.shape[0], -1)).\
            swapaxes(2,1).reshape((batch, ) + self.prev.shape)
            
        return output

    def get_param_grad(self):
        batch = self.delta.shape[0]
    
        swapim = self.input.swapaxes(0,1)
        col = batch_im2col(swapim, self.delta.shape[2:], self.stride)\
            .swapaxes(1,2)
        
        swapdel = self.delta.swapaxes(0,1).reshape(self.delta.shape[1], -1)
        
        dw = np.inner(col, swapdel).swapaxes(2,1).reshape(self.kernels.shape)
        
        db = swapdel.sum(axis=1)
        
        return dw/batch, db/batch

    def SGDtrain(self, rate, **kwargs):
        k_update, b_update = self.get_param_grad()
        self.kernels -= rate * k_update
        self.bias -= rate * b_update

#    def L2train(self, rate, reg):
#        k_update, b_update = self.get_param_grad()
#        self.kernels -= rate * k_update +\
#                        self.kernels * (rate * reg) / len(self.delta)

    def __str__(self):
        res = lm.AbstractLayer.__str__(self)
        res += '   ->   kernels: {}'.format(self.kernels.shape)
        return res


class max_pool(lm.AbstractLayer):

    def __init__(self, pool_shape=(2, 2), stride=(2, 2), shape=None, **kwargs):
        
        assert (shape is None) ^ (pool_shape is None),\
            "'pool_shape=' XOR 'shape=' must be defined"
      
        '''aggfunc will be applied to pools aggregating their values to one
           restriction is, that it should be able to applied np.arrays and
           have argument "axis="
        '''
        
        if type(pool_shape) == int:
            pool_shape = (pool_shape, pool_shape)
        
        if type(stride) == int:
            stride = (stride, stride)
        
        lm.AbstractLayer.__init__(self, shape=shape, type='max pool', **kwargs)
        if self.prev:
            if shape:
                sp = np.divide(self.prev.shape, shape)
                '''First dimension is the number of feature maps in the previous
                   layer'''
                self.pool_shape = tuple(sp[1:])
                self.stride = self.pool_shape
            else:
                self.pool_shape = pool_shape
                self.stride = stride
                C, N, M = self.prev.shape
                n, m = self.pool_shape
                s1, s2 = self.stride
                ver = (N-n)/s1 + 1
                hor = (M-m)/s2 + 1
                self.shape = (C, ver, hor)
                self.width = np.prod(self.shape)

    def get_local_output_inference(self, input):

        if len(input.shape) == 3:
            'if input is a single sample, extend it to a 1 sized batch'
            x = np.expand_dims(input, 0)
        else:
            x = input
        
        batch = len(x)   
        '''Channels are not pooled,
           transform to (batch*channel, 1, ...) shape'''
        x = x.reshape(-1, 1, *x.shape[2:])
        col = batch_im2col(x, self.pool_shape, self.stride)
        
        out = col.max(axis=1)
        
        return out.reshape((batch,) + self.shape)

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
