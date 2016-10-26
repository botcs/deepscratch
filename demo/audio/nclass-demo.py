#!/usr/bin/ipython
## This is an example of a simple sound capture script.
##
## The script opens an ALSA pcm for sound capture. Set
## various attributes of the capture, and reads in a loop,
## Then prints the volume.
##
## To test it out, run it and shout at your microphone:

import alsaaudio, time, audioop, sys, os
import numpy as np
import matplotlib.pyplot as plt
import network_module as nm
import cPickle
from subprocess import call
N_CLASS = 4


SAMPLE_RATE = 8000
NFFT = 1000
# How many sample frame should be concatenated in recording
# if record window is 0 then recording stops when the kick_out level is reached
RECORD_WINDOW = 0
RECORD = np.array([])
records = []
# recording kicks in, when kick_in times the mean noise level is reached
kick_in = 5
# if RECORD_WINDOW is set to 0 recording stops when kick_out times the mean noise level is reached
kick_out = 1

# if maximum silent samples is reached recording will stop
max_ss = 10

silent_samples = 0

# For demonstrating purposes the training sample gathering should be ended with a clap, which is simply recognized as a large spike in the input signal - therefore if the maximum magnitude reaches a critically high level the capturing ends.
clap_treshold = 1000000000

# Open the device in nonblocking capture mode. The last argument could
# just as well have been zero for blocking mode. Then we could have
# left out the sleep call in the bottom of the loop
inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,alsaaudio.PCM_NONBLOCK)

# Set attributes: Mono, 8000 Hz, 16 bit little endian samples
inp.setchannels(1)
inp.setrate(8000)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
# The period size controls the internal number of frames per period.
# The significance of this parameter is documented in the ALSA api.
# For our purposes, it is suficcient to know that reads from the device
# will return this many frames. Each frame being 2 bytes long.
# This means that the reads below will return either 320 bytes of data
# or 0 bytes of data. The latter is possible because we are in nonblocking
# mode.
frames = 100
inp.setperiodsize(frames)
N = 500
maxes = np.zeros(N)
recording = False
pre_loaded = False
# Noise statistics parameter initialization
mean = 0
sum_of_max = 0
i = 0

pre_load = False
if len(sys.argv) > 1:
    global records
    global pre_load
    
    fname = sys.argv[1]
    print 'The following argument was passed', fname
    print 'Loading trainset from ./' + fname
    assert os.path.exists(fname), 'File does not exist: ' + fname
    records = cPickle.load(file(fname, 'r'))
    assert type(records) == list and type(records[0]) == np.ndarray,\
        'Wrong trainset loaded, or ' + fname + ' is corrupted.'
    print 'Trainset loaded, # of total train samples: ', len(records)
    pre_load = True
    
    
else:
    # If no arguments were passed then the trainset is going to be recorded live.
    def stop_record():
        global RECORD
        global recording
        global records
        global silent_samples
        records.append(RECORD)
        silent_samples = 0
        recording = False                
        print "\nRecording finished, number of total records:", len(records)

    try:
        
        while True:
            # Read data from device
            l, data = inp.read()    
            if l:
                x = np.fromstring(data, dtype=np.int16)
                i += 1   
                m = max(abs(x))
                maxes[i%N] = m
                
                if i > N:
                #gathered noise stat
                    if m > clap_treshold * sum_of_max / i: 
                        break
                        
                    if m >  kick_in * sum_of_max / i:
                        if not recording: print i, "RECORDING", len(x)
                        RECORD = np.array([])
                        recording = True
                elif i == N: 
                    print 'Done!'
                    print 'waiting for k training samples for N class...'
                else:
                    
                    print '\rGathering environmental noise statistics...', 
                    sys.stdout.flush()
                    
                # the noise introduced by the training samples' is not significant
                # for short demonstrations
                if not recording: sum_of_max += m
                else: sum_of_max += sum_of_max / i
                
                if RECORD_WINDOW > 0:
                    # FIXED TIME FRAME CAPTURING
                    if recording:
                        if len(RECORD) < RECORD_WINDOW * frames:
                            RECORD = np.concatenate([RECORD, x])
                            print '\rconcatenated, current length:', len(RECORD),
                            sys.stdout.flush()
                        else: stop_record()
                        
                else:
                    # When the largest magnitude falls below kick_out recording is stopped
                    if recording:
                        if m > kick_out * sum_of_max / i:
                            RECORD = np.concatenate([RECORD, x])
                            print '\rconcatenated, current length:', len(RECORD),
                            sys.stdout.flush()
                        elif silent_samples < max_ss: 
                            silent_samples += 1
                        else:
                            stop_record()
                          

                    
            if i%(N/20) == 0:
                plt.clf()
                plt.plot(maxes)  
                plt.pause(.01) 
    except KeyboardInterrupt:
        plt.close()


#########################
# PREPROCESSING INPUTS
fft_records = np.abs([np.fft.rfft(r, n=2 * NFFT) for r in records])
fft_records /= fft_records.max(axis=1)[:, None]

labels = np.eye(N_CLASS)
labels = np.repeat(labels, len(records)/N_CLASS, axis=0)
train = (fft_records, labels)



print 'Constructing network...'
#########################
# NETWORK DEFINITION
nn = nm.network(in_shape=train[0].shape[1], criterion='softmax')
nn.add_full(N_CLASS, sharp=True)
#########################
print nn


nn.SGD(train_policy=nn.fix_epoch, training_set=train, L2=True,
    batch=1, rate=0.05, reg=0.05, epoch=150)

print "TRAINING FINISHED"
print "Network performance on trainset: ", nn.test_eval(train), 'of', len(train[0]) 

def test_action():
    print "\n\nRecording finished, network's output"
    np.abs(np.fft.rfft(RECORD, 2 * NFFT))
    fft_online = np.abs(np.fft.rfft(RECORD, 2 * NFFT))
    fft_online /= fft_online.max()
    nn_assumption = nn.get_output(fft_online)
    nn_assumption = nn_assumption.squeeze()
    print nn_assumption
    print 'Most likely class: ', nn_assumption.argmax()+1

    print("Confidence %.2f " % (nn_assumption.max() * 100))
    print "CLAP IF THE ASSUMPTION WAS CORRECT!\n\n"
    
    nna = nn_assumption.argmax()
    
    if nna == 0:
        os.system("eog cough.jpg &")
    elif nna == 1:
        os.system("eog snap.jpg &")
    elif nna == 2:
        os.system("eog pop.jpg &")
    

try:
    while True:
        # Read data from device
        l, data = inp.read()    
                
        if l:
            x = np.fromstring(data, dtype=np.int16)
            i += 1   
            m = max(abs(x))
            maxes[i%N] = m
            
            if i > N:
            #gathered noise stat
                if m > clap_treshold * sum_of_max / i: 
                    break
                    
                if m >  kick_in * sum_of_max / i:
                    if not recording: print i, "RECORDING", len(x)
                    RECORD = np.array([])
                    recording = True
            elif i == N: 
                print 'Done!'
                print 'Waiting for test samples... '
            else:
                
                print '\rGathering environmental noise statistics...', 
                sys.stdout.flush()

            if not recording: sum_of_max += m
            else: sum_of_max += sum_of_max / i
            
            if m > clap_treshold * sum_of_max / i and not recording: 
                np.concatenate(train[0], fft_record[None, :])
                np.concatenate(train[1], nn_assumption[None, :])
                nn.last_epoch = 0
                nn.SGD(train_policy=nn.fix_epoch, training_set=train, L2=True,
                    batch=len(train[0]), rate=0.05, reg=0.005, epoch=3)
                
            
            if RECORD_WINDOW > 0:
                # FIXED TIME FRAME CAPTURING
                if recording:
                    if len(RECORD) < RECORD_WINDOW * frames:
                        RECORD = np.concatenate([RECORD, x])
                        print '\rconcatenated, current length:', len(RECORD),
                        sys.stdout.flush()
                    elif silent_samples < max_ss: 
                        silent_samples += 1
                    else:
                        test_action()  
                        recording = False
                        silent_samples = 0                  
                    
            else:
                # When the largest magnitude falls below kick_out recording is stopped
                if recording:
                    if m > kick_out * sum_of_max / i:
                        RECORD = np.concatenate([RECORD, x])
                        print '\rconcatenated, current length:', len(RECORD),
                        sys.stdout.flush()
                    elif silent_samples < max_ss: 
                        silent_samples += 1
                    else:
                        test_action()
                        recording = False 
                        silent_samples = 0                       

                    
        if i%(N/20) == 0:
            plt.clf()
            plt.plot(maxes)  
            plt.pause(.01) 
except KeyboardInterrupt:
    plt.close()

print "TESTING FINISHED, plotting Network Statistics... "

def freq_weight_plot():
    for i in range(N_CLASS):
        plt.subplot(3, N_CLASS, i + 1)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)    
        plt.plot(records[i * len(train[0]) / N_CLASS])
        
    
    for i in range(N_CLASS):
        plt.subplot(3, N_CLASS, i + 1 + N_CLASS)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)  
        plt.plot(fft_records[i * len(train[0]) / N_CLASS])
    

    for i in range(N_CLASS):
        plt.subplot(3, N_CLASS, i + 1 + 2 * N_CLASS)    
        plt.gca().get_xaxis().set_visible(False)    
        plt.plot(nn[1].weights[i])

    plt.show()

if '-plot' in sys.argv: freq_weight_plot()    

if not pre_load:
    print "Saving to ./autosave/{}.aud".format(id(records))
    cPickle.dump(records, file('./autosave/' + str(id(records)) + '.aud', 'w'))


