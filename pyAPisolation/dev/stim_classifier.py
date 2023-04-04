#######
# This experimental script will be used to generate a LSTM model for classifying stimuli
# Out of frustration for edge cases, I will be using a simple LSTM model
#######

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from scipy.signal import resample, decimate
STIM_ENCODING = {'long_square': 0, 'short_square': 1, 'ramp': 2, 'sine': 3}
STIM_DECODING = {0: 'long_square', 1: 'short_square', 2: 'ramp', 3: 'sine'}

def create_stimuli_waveform(x, length, type='long_square', test_pulse=False):


    if type == 'long_square':
        x = create_long_square_waveform(x, length)
    elif type == 'short_square':
        x = create_short_square_waveform(x, length)
    elif type == 'ramp':
        x = create_ramp_waveform(x, length)
    elif type == 'sine':
        x = create_sine_waveform(x, length)

    #add a test pulse if test_pulse is True
    #test_pulse will be be a random pulse of -1 to 1 with a length of 1/10th the length of the stimulus
    if test_pulse:
        x = add_test_pulse(x, length)
    return x

def create_dataset():
    #create a dataset of 1000 waveforms of lengths 3000 to 5000
    #each waveform will be a random waveform of long_square, short_square, ramp, sine
    #each waveform will randomly have a test pulse
    #each waveform will be a random length between 1000 and 5000
    dataset = []
    for i in range(100000):
        length = np.random.randint(3000, 10000)
        x = np.zeros(length)
        length_stim = np.random.randint(1000, length//2)
        type = np.random.choice(['long_square', 'short_square', 'ramp', 'sine'])
        test_pulse = np.random.choice([True, False])
        x = create_stimuli_waveform(x, length_stim, type, test_pulse)
        dataset.append([x, STIM_ENCODING[type]])

    #pack the labels
    labels = torch.tensor([x[1] for x in dataset], dtype=torch.long).reshape(-1, 1)
    #pad and pack the sequences
    dataset = torch.nn.utils.rnn.pad_sequence([torch.tensor(x[0]) for x in dataset], batch_first=True)
    #dataset = torch.nn.utils.rnn.pack_padded_sequence(dataset, [len(x) for x in dataset], batch_first=True, enforce_sorted=False)
    
    return dataset, labels


def add_test_pulse(x, length):
    #create a random pulse of -1 to 1 at a random index
    #the testpulse will be in a new array that is at most 4000 samples long and affixed to the start
    x_test = np.zeros(4000)
    idx = np.random.randint(0, len(x_test))
    x_test[idx:idx+int(length/10)] = np.random.uniform(-1, 1, size=1)[0]
    x = np.concatenate((x_test, x))
    return x

def create_long_square_waveform(x, length):
    #create a np.zeros array of length
    y = np.zeros(len(x))
    #create a random pulse of -1 to 1 at a random index
    idx = np.random.randint(0, len(x) - length)
    y[idx:idx+length] = np.random.uniform(-1, 1, size=1)[0]
    return y

def create_short_square_waveform(x, length):
    #create a np.zeros array of length
    y = np.zeros(len(x))
    #create a random pulse of -1 to 1 at a random index
    #short square will be a random length between 1/10th and 1/5 the length of the stimulus
    length = np.random.randint(length//10, length//5)
    idx = np.random.randint(0, len(x) - length)
    y[idx:idx+length] = np.random.uniform(-1, 1, size=1)[0]
    return y

def create_ramp_waveform(x, length):
    #create a np.zeros array of length
    y = np.zeros(len(x))
    #create a random pulse of -1 to 1 at a random index
    idx = np.random.randint(0, len(x) - length)
    start_val = np.random.uniform(-1, 1, size=1)[0]
    y[idx:idx+length] = np.linspace(start_val, start_val*-1, length)
    return y

def create_sine_waveform(x, length):
    #create a np.zeros array of length
    y = np.zeros(len(x))
    #create a random pulse of -1 to 1 at a random index
    idx = np.random.randint(0, len(x) - length)
    #sine will be a random frequency between 1/10th and 1/5 the length of the stimulus
    freq = np.random.randint(length//10, length//5)
    y[idx:idx+length] = np.sin(np.linspace(0, freq*np.pi, length))
    return y

def plot_examples(dataset, labels, num_examples=8):
    #plot examples of the dataset as a grid
    #dataset is a list of np.arrays
    #labels is a list of labels
    #num_examples is the number of examples to plot
    subplots = int(np.ceil(np.sqrt(num_examples)))
    fig, ax = plt.subplots(subplots, subplots, figsize=(10, 10))

    for i in range(num_examples):
        ax[i//subplots, i%subplots].plot(dataset[i])
        ax[i//subplots, i%subplots].set_title(STIM_DECODING[labels[i].detach().numpy()[0]])
        ax[i//subplots, i%subplots].set_xticks([])
        ax[i//subplots, i%subplots].set_yticks([])
        ax[i//subplots, i%subplots].set_ylim([-1.1, 1.1])
    plt.show()

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):

        #set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        x, _status = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc(x)
        return x

    def init_hidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size),
                torch.zeros(self.num_layers, 1, self.hidden_size))

class dense(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(dense, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for i in range(num_layers-1)])
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        
        for i in range(self.num_layers-1):
            x = self.fcs[i](x)
            x = torch.nn.functional.relu(x)
        
        x = self.fc2(x)

        x = torch.nn.functional.softmax(x, dim=1)
        return x

def create_LSTM_model(input_size, hidden_size, output_size, num_layers=1):
    model = LSTM(input_size, hidden_size, output_size, num_layers)
    return model

def create_dense_model(input_size, hidden_size, output_size, num_layers=1):
    model = dense(input_size, hidden_size, output_size, num_layers)
    return model

def train_model():
    #create a dataset
    dataset, labels = create_dataset()
    full_dataset = torch.utils.data.TensorDataset(dataset, labels)
    plot_examples(dataset, labels)
    #make batch size of 32
    train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=32, shuffle=True)
    #create a LSTM model
    model = create_dense_model(dataset.shape[1], 1024, 4, 3)
    #Cosine annealing scheduler
    #create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=0, last_epoch=-1)

    #loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    #accuracy function
    accuracy = []
    #train the model
    i = 0
    for x, y in train_loader:
        
        optimizer.zero_grad()
        x = torch.tensor(x).float()
        y_pred = model(x)
        loss = loss_fn(y_pred, y.reshape(-1))
        #print(loss)
        loss.backward()
        optimizer.step()
        #calculate accuracy
        pred = torch.argmax(y_pred, dim=1)
        accuracy.append(torch.sum(pred == y.reshape(-1))/len(y))
        i += 1
        if i % 100 == 0:
            scheduler.step()
            #print accuracy
            print('accuracy: ', np.mean(accuracy), 'loss', loss.detach().numpy())
            accuracy = []

    #return the model
    return model


class stimClassifier():
    def __init__(self, model_path='model.pt'):
        self.model = create_dense_model(13999, 1024, 4, 3)
        self.model.load_state_dict(torch.load(os.path.dirname(__file__) + '/' + model_path))
        self.model.eval()
    
    def predict(self, x):
        x = self._reshape_x(x)
        x = torch.tensor(x).float()
        y_pred = self.model(x)
        pred = torch.argmax(y_pred, dim=1)
        return pred.detach().numpy()
    
    def predict_proba(self, x):
        x = self._reshape_x(x)
        x = torch.tensor(x).float()
        y_pred = self.model(x)
        return y_pred.detach().numpy()
    
    def fit(self, x, y):
        pass

    def score(self, x, y):
        x = self._reshape_x(x)
        x = torch.tensor(x).float()
        y_pred = self.model(x)
        pred = torch.argmax(y_pred, dim=1)
        return torch.sum(pred == y.reshape(-1))/len(y)
    
    def fit_transform(self, x, y):
        pass

    def _reshape_x(self, x):
        #we need to reshape x to be the correct size,
        #infer the correct size from the model
        x = resample(x, 13999, axis=1)
        #add a batch dimension
        #x = x[np.newaxis, :]
        #rescale to -1, 1
        x = np.apply_along_axis(lambda i: i/np.max(np.abs(i)), 1, x)
        return x



if __name__ == '__main__':
    #model = train_model()
    #save the model
    #torch.save(model.state_dict(), 'model.pt')

    #load the model
    model = stimClassifier()
    #create a dataset
    dataset, labels = create_dataset()
    #predict on the dataset
    pred = model.predict(dataset)
    #calculate accuracy
    print('accuracy: ', np.sum(pred == labels.reshape(-1).detach().numpy())/len(labels))
