import sys
import os
import warnings
import numpy as np

def im2col(input2d, block_size, skip):
    #source http://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
    
#    A = np.random.randint(0,9,(2,4,4)) # Sample input array
#                    
#    B = [2,2] # Sample blocksize (rows x columns)
#    skip=[2,2]

    A = input2d
    B = block_size
    # Parameters 
    D,M,N = A.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get Starting block indices
    start_idx = np.arange(B[0])[:,None]*N + np.arange(B[1])

    # Generate Depth indeces
    didx=M*N*np.arange(D)
    start_idx=(didx[:,None]+start_idx.ravel()).reshape((-1,B[0],B[1]))

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    out = np.take (A,start_idx.ravel()[:,None] + offset_idx[::skip[0],::skip[1]].ravel())
    
    return out

import numpy as np
def batch_im2col(input2d, block_size, skip):
    # inspired by:
    # http://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
    
#    A = np.random.randint(0,9,(2,4,4)) # Sample input array
#                    
#    B = [2,2] # Sample blocksize (rows x columns)
#    skip=[2,2]

    A = input2d
    B = block_size
    
    # Parameters 
    batch, D,M,N = A.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get batch block indices
    batch_idx = np.arange(batch)[:, None, None] * D * M * N
    
    # Get Starting block indices
    start_idx = np.arange(B[0])[None, :,None]*N + np.arange(B[1])

    # Generate Depth indeces
    didx=M*N*np.arange(D)
    start_idx=(didx[None, :, None]+start_idx.ravel()).reshape((-1,B[0],B[1]))

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[None, :, None]*N + np.arange(col_extent)[None, :]

    # Get all actual indices & index into input array for final output
    act_idx = (batch_idx + 
        start_idx.ravel()[None, :, None] + 
        offset_idx[:,::skip[0],::skip[1]].ravel())
    
    out = np.take (A, act_idx)
    
    return out


def ensure_dir(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError as e:
            print ('Cannot make directory: ' + str(e))
            raise



def warning(message, instance, warn_type=FutureWarning):
    note = '\n  Warning for: ' + repr(instance)
    warnings.warn(message + note, warn_type)


class StatusBar:

    def __init__(self, total, barLength=30):
        self.total = total
        self.curr = 0
        self.percentage = 0
        self.barLength = barLength

    def barStr(self):
        currBar = self.barLength * self.percentage / 100
        return '[' + "=" * currBar + " " * (self.barLength - currBar) + ']'

    def printBar(self, msg):
        if(self.percentage <= 100):
            print("\r  " + self.barStr() + " (" +
                  str(self.curr) + '/' + str(self.total) + ")   " +
                  str(100 * self.curr / self.total) +
                  "%  {}   ".format(msg)),
            sys.stdout.flush()
            if(self.percentage == 100):
                print '\n'

    def update(self, msg):
        self.curr += 1
        currPercentage = self.curr * 100 / self.total
        if(currPercentage > self.percentage):
            self.percentage = currPercentage
            self.printBar(msg)
