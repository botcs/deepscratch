#!/usr/bin/ipython
## This is an example of a simple sound capture script.
##
## The script opens an ALSA pcm for sound capture. Set
## various attributes of the capture, and reads in a loop,
## Then prints the volume.
##
## To test it out, run it and shout at your microphone:

import alsaaudio, time, audioop, sys
import numpy as np
import matplotlib.pyplot as plt
import network_module as nm

SAMPLE_RATE = 8000

# How many sample frame should be concatenated in recording
RECORD_WINDOW = 100
RECORD = np.array([])
records = []

records = []
records.append(
  np.sin(20*np.linspace(0,100, 12000)) + 
  np.sin(50*np.linspace(0,100, 12000)) +
  np.sin(5*np.linspace(0,100, 12000))
)

records.append(
  np.sin(20*np.linspace(0,100, 12000)) + 
  np.sin(50*np.linspace(0,100, 12000)) +
  np.sin(40*np.linspace(0,100, 12000))
)

print records

fft_records = np.abs([np.fft.rfft(r, n=10000) for r in records])
fft_records /= fft_records.max(axis=1)[:, None]


print 'Constructing network...'
#########################
# NETWORK DEFINITION

nn      = nm.network(in_shape=fft_records.shape[1], criterion='softmax')
#N05     = nm.network(in_shape=fft_records.shape[1], criterion='softmax')
#N01     = nm.network(in_shape=fft_records.shape[1], criterion='softmax')
#N001    = nm.network(in_shape=fft_records.shape[1], criterion='softmax')
#N0001   = nm.network(in_shape=fft_records.shape[1], criterion='softmax')
#N0      = nm.network(in_shape=fft_records.shape[1], criterion='softmax')

nn.add_full(2, sharp=True)
#N05.add_full(2, sharp=True)
#N01.add_full(2, sharp=True)
#N001.add_full(2, sharp=True)
#N0001.add_full(2, sharp=True)
#N0.add_full(2, sharp=True)
#########################

labels = np.zeros((len(records), 2))
labels[:len(records)/2, 0] = 1
labels[len(records)/2:, 1] = 1
train = (fft_records, labels)
nn.SGD(train_policy=nn.fix_epoch, training_set=train, L2=True,
    batch=1, rate=0.05, reg=0.5, epoch=100)
#N05.SGD(train_policy=N05.fix_epoch, training_set=train, L2=True,
#    batch=1, rate=0.05, reg=0.5, epoch=5)
#N01.SGD(train_policy=N01.fix_epoch, training_set=train, L2=True,
#    batch=1, rate=0.05, reg=0.1, epoch=5)
#N001.SGD(train_policy=N001.fix_epoch, training_set=train, L2=True,
#    batch=1, rate=0.05, reg=0.01, epoch=5)
#N0001.SGD(train_policy=N0001.fix_epoch, training_set=train, L2=True,
#    batch=1, rate=0.05, reg=0.001, epoch=5)
#N0.SGD(train_policy=N0.fix_epoch, training_set=train, L2=False,
#    batch=1, rate=0.05, reg=0.0, epoch=5)
        

print "TRAINING FINISHED, waiting for test inputs"

def freq_weight_plot():
    plt.subplot(321)
    plt.xlabel('class I. sample (x: time)')
    plt.plot(records[0][:200])


    plt.subplot(322)
    plt.xlabel('class II. sample (x: time)')
    plt.gca().get_yaxis().set_visible(False)
    plt.plot(records[-1][:200])


    plt.subplot(323)
    plt.xlabel('class I. FFT (x: freq)')
    plt.plot(fft_records[0])

    plt.subplot(324)
    plt.xlabel('class II. FFT (x: freq)')
    plt.gca().get_yaxis().set_visible(False)
    plt.plot(fft_records[-1])


    plt.subplot(325)
    plt.xlabel("class I. recognizing neuron's weight plot (x: corresponding freq)")
    plt.plot(nn[1].weights[0])

    plt.subplot(326)
    plt.gca().get_yaxis().set_visible(False)
    plt.xlabel("class I. recognizing neuron's weight plot (x: corresponding freq)")
    plt.plot(nn[1].weights[1])

    plt.show()

def L_plot():
    plt.subplot(721)
    plt.xlabel('class I. sample (x: time)')
    plt.plot(records[0][:100])

    plt.subplot(722)
    plt.xlabel('class II. sample (x: time)')
    plt.plot(records[-1][:100])

    plt.subplot(723)
    plt.xlabel('class I. FFT (x: freq)')
    plt.plot(fft_records[0])
    
    plt.subplot(724)
    plt.xlabel('class II. FFT (x: freq)')
    plt.plot(fft_records[-1])

    plt.subplot(725)
    plt.plot(N05[1].weights[0])
    
    plt.subplot(726)
    plt.xlabel("L2 regularization factor = 0.5")
    plt.plot(N05[1].weights[1])


    plt.subplot(727)
    plt.plot(N01[1].weights[0])
    
    plt.subplot(728)
    plt.xlabel("L2 regularization factor = 0.1")
    plt.plot(N01[1].weights[1])

    plt.subplot(729)
    plt.plot(N001[1].weights[0])
    
    plt.subplot(7,2,10)
    plt.xlabel("L2 regularization factor = 0.01")
    plt.plot(N001[1].weights[1])

    plt.subplot(7,2,11)
    plt.plot(N0001[1].weights[0])
    
    plt.subplot(7,2,12)
    plt.xlabel("L2 regularization factor = 0.001")
    plt.plot(N0001[1].weights[1])

    plt.subplot(7,2,13)
    plt.plot(N0[1].weights[0])

    plt.subplot(7,2,14)
    plt.xlabel("L2 regularization factor = 0")
    plt.plot(N0[1].weights[1])
    
    plt.show()

freq_weight_plot()
    
print("End...")

