#!/usr/bin/ipython
import pyaudio
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import network_module as nm
import cPickle
import csv as csv

####################################
## HIGH LEVEL, 
## MAIN HYPER PARAMETHERS
####################################
# Number of classes of samples
N_CLASS = 3
# Precision of pre-processing
NFFT = 1000
# Confidence threshold for action calls
CONF_THRESHOLD = 0.6


####################################
## NEURAL NETWORK HYPERPARAMETHERS
####################################
def init_network():
    print 'Constructing network...'
    global nn
    #########################
    # NETWORK DEFINITION
    nn = nm.network(in_shape=train[0].shape[1], criterion='softmax')
    #nn.add_full(100, sharp=True)
    #nn.add_activation('relu')
    nn.add_full(N_CLASS, sharp=True)
    #########################
    print nn
    
    nn.SGD(
        train_policy=nn.fix_epoch, 
        training_set=train, 
        L2=True,
        batch=1, 
        rate=0.05, 
        reg=0.1, 
        epoch=150, 
        epoch_call_back=print_state)





####################################
## MID LEVEL, 
## TRIGGERING PARAMETHERS
####################################
# How many sample frame should be concatenated in recording
# if record window is 0 then recording stops when the kick_out level is reached
RECORD_WINDOW = 0
# recording kicks in, when kick_in * mean noise level is reached
kick_in = 10
# if RECORD_WINDOW is set to 0 recording stops when kick_out times the mean noise level is reached
kick_out = 1
# if number of silent samples is reached recording will stop
silent_samples = 3
# For demonstrating purposes the training sample gathering should be ended with a clap, which is simply recognized as a large spike in the input signal - therefore if the maximum magnitude reaches a critically high level the capturing ends.
clap_treshold = 1000000000



####################################
## ACTION CALLS
## TRIGGERED IN TEST
####################################
def read_actions():
    with open('action_calls', 'r') as f:
        calls = [r[0] for r in csv.reader(f)]
    return calls


def test_action(RECORD):
    print "\n\nRecording finished, network's output"
    np.abs(np.fft.rfft(RECORD, 2 * NFFT))
    fft_online = np.abs(np.fft.rfft(RECORD, 2 * NFFT))
    fft_online /= fft_online.max()
    nn_assumption = nn.get_output(fft_online).squeeze()
    
    choice = nn_assumption.argmax()
    confidence = nn_assumption.max()
    
    print nn_assumption
    print 'Most likely class: ', choice+1

    print("Confidence %.2f " % (confidence * 100))
    #print "CLAP IF THE ASSUMPTION WAS CORRECT!\n\n"
    
    if confidence > CONF_THRESHOLD:
        # shell commands called, defined in action_calls
        
        # UNCOMMENT TO REPARSE ACTIONS UPON EVERY CALL
        calls = read_actions()
        os.system(calls[choice])
    else:
        os.system('eog confused.png &')
    

####################################
## LOW LEVEL, SAMPLING PARAMETHERS
## SAMPLER INITIALIZATION
####################################
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
AVG_LEN = 500
####################################




####################################
## FILE AND STATISTICS INITIALIZATION
## TRAIN SAMPLE RECORDING
####################################
RECORD = np.array([])
records = []
curr_silsamples = 0

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16, 
    channels=1, 
    rate=SAMPLE_RATE,
    input=True, 
    output=True,
    frames_per_buffer=BUFFER_SIZE)

recording = False
pre_load = False

# Noise statistics parameter initialization
mean = 0
sum_of_max = 0
i = 0
maxes = np.zeros(AVG_LEN)
treshold_line = np.ones(AVG_LEN)

calls = read_actions()
    
if len(sys.argv) > 1:
    # pre-loaded samples
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
        global curr_silsamples
        records.append(RECORD)
        curr_silsamples = 0
        recording = False                
        print "\nRecording finished, number of total records:", len(records)

    try:
        
        while True:
            # Read data from device
            data = stream.read(BUFFER_SIZE)
            if not data:
                # if data could not be read
                continue
            x = np.fromstring(data, dtype=np.int16)
            i += 1   
            m = max(abs(x))
            maxes[i%AVG_LEN] = m
            
            if i > AVG_LEN:
            #gathered noise stat
                if m > clap_treshold * sum_of_max / i: 
                    break
                    
                if m >  kick_in * sum_of_max / i:
                    if not recording: print i, "RECORDING", len(x)
                    RECORD = np.array([])
                    recording = True
            elif i == AVG_LEN: 
                print 'Done!'
                print 'waiting for k training samples for N class...'
            else:
                print '\rGathering environmental noise statistics...', 
                sys.stdout.flush()
                
            if not recording: sum_of_max += m
            else: sum_of_max += sum_of_max / i
            
            if RECORD_WINDOW > 0:
                # FIXED TIME FRAME CAPTURING
                if recording:
                    if len(RECORD) < RECORD_WINDOW * BUFFER_SIZE:
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
                    elif curr_silsamples < silent_samples: 
                        curr_silsamples += 1
                    else:
                        stop_record()
                          
            if i%(10) == 0:
                plt.clf()
                plt.plot(maxes)  
                plt.plot(kick_in * sum_of_max / i * treshold_line)
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


def print_state():
    print ' --- epoch ', nn.last_epoch, ' --- learned ', nn.test_eval(train), 'of', len(train[0])  

init_network()    

print "TRAINING FINISHED"
print "Network final performance on trainset: ", nn.test_eval(train), 'of', len(train[0]) 




try:
    while True:
        # Read data from device
        data = stream.read(BUFFER_SIZE)
        if not data:
            # if data could not be read
            continue
    
        x = np.fromstring(data, dtype=np.int16)
        i += 1   
        m = max(abs(x))
        maxes[i%AVG_LEN] = m
        
        if i > AVG_LEN:
        #gathered noise stat
            if m > clap_treshold * sum_of_max / i: 
                break
                
            if m >  kick_in * sum_of_max / i:
                if not recording: print i, "RECORDING", len(x)
                RECORD = np.array([])
                recording = True
        elif i == AVG_LEN: 
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
                if len(RECORD) < RECORD_WINDOW * BUFFER_SIZE:
                    RECORD = np.concatenate([RECORD, x])
                    print '\rconcatenated, current length:', len(RECORD),
                    sys.stdout.flush()
                elif curr_silsamples < silent_samples: 
                    curr_silsamples += 1
                else:
                    test_action(RECORD)  
                    recording = False
                    curr_silsamples = 0                  
                
        else:
            # When the largest magnitude falls below kick_out recording is stopped
            if recording:
                if m > kick_out * sum_of_max / i:
                    RECORD = np.concatenate([RECORD, x])
                    print '\rconcatenated, current length:', len(RECORD),
                    sys.stdout.flush()
                elif curr_silsamples < silent_samples: 
                    curr_silsamples += 1
                else:
                    test_action(RECORD)
                    recording = False 
                    curr_silsamples = 0                       

                    
        if i%(10) == 0:
            plt.clf()
            plt.plot(maxes)  
            plt.plot(kick_in * sum_of_max / i * treshold_line)
            plt.pause(.01) 

except KeyboardInterrupt:
    plt.close()


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

if not pre_load or '-plot' in sys.argv: 
    print "TESTING FINISHED, plotting Network Statistics... "
    freq_weight_plot()    

if not pre_load:
    file_name = './autosave/{}_{}.aud'.format(N_CLASS, id(records))
    os.system('mkdir autosave')
    print "Saving to", file_name
    cPickle.dump(records, file(file_name, 'w'))

os.system('eog exit.gif -f &')

