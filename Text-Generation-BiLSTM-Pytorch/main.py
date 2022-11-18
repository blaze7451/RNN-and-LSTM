import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
from src import TextGenerator
from utils import Preprocessing
from utils import parameter_parser

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Execution():
    def __init__(self, args):
        self.file='C:\\Python\\Pytorch\\RNN and LSTM\\Text-Generation-BiLSTM-Pytorch\\data\\Atlas Shrugged (Ayn Rand)1.txt'
        self.window=args.window
        self.batch_size=args.batch_size
        self.learning_rate=args.learning_rate
        self.num_epochs=args.num_epochs

        self.targets = None
        self.sequences = None
        self.vocab_size = None
        self.char_to_idx = None
        self.idx_to_char = None

    def prepare_data(self):

        # Initialize preprocessor object
        preprocessing=Preprocessing()

        # The 'file' is loaded and split by char
        text=preprocessing.read_dataset(self.file)

        # Given 'text', it is created two dictionaries
        # a dictiornary about: from char to index, from index to char
        self.char_to_idx, self.idx_to_char=preprocessing.create_dictionary(text)

        # Given the 'window', it is created the set of training sentences as well as the set of target chars
        self.sequences, self.targets=preprocessing.build_sequence_target(text, self.char_to_idx, window=self.window)

        # Gets the vocabuly size
        self.vocab_size=len(self.char_to_idx)

    def train(self, args):

        # Model initialization
        model=TextGenerator(args, self.vocab_size).to(device)

        # Optimizer initialization
        optimizer=optim.AdamW(model.parameters(), lr=self.learning_rate)

        # Defining number of batches
        num_batches = int(len(self.sequences) / self.batch_size)

        # Set model in training mode
        model.train()

        #Training phase
        for epoch in range(self.num_epochs):

            #Mini batches
            for i in range(num_batches):

                #Batch definition
                try:
                    x_batch=self.sequences[i*self.batch_size:(i+1)*self.batch_size]
                    y_batch=self.targets[i*self.batch_size:(i+1)*self.batch_size]

                except:
                    x_batch=self.sequences[i*self.batch_size:]
                    y_batch=self.targets[i*self.batch_size:]

                # Convert numpy array into torch tensors
                x=torch.from_numpy(x_batch).type(torch.LongTensor).to(device)
                y=torch.from_numpy(y_batch).type(torch.LongTensor).to(device)

                #Feed the model
                y_pred=model(x).to(device)

                #loss calculation
                loss=F.cross_entropy(y_pred, y.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Epoch: {}, loss={:.5f}".format(epoch+1, loss.item()))

        torch.save(model.state_dict(), 'C:\\Python\\Pytorch\\RNN and LSTM\\Text-Generation-BiLSTM-Pytorch\\weights\\textGenerator_model.pt')

    @staticmethod
    def generator(model, sequences, idx_to_char, n_chars):

        # Set the model in evalulation mode
        model.eval()

        # Define the softmax function
        softmax=nn.Softmax(dim=1)

        # Randomly is selected the index from the set of sequences
        start = np.random.randint(0, len(sequences)-1)

        # The pattern is defined given the random idx
        pattern = sequences[start]

        # By making use of the dictionaries, it is printed the pattern
        print("\nPattern: ")
        print(''.join([idx_to_char[value] for value in pattern]))

        # In full_prediction we will save the complete prediction
        full_prediction=torch.Tensor(pattern.copy())
        pattern=torch.Tensor(pattern)
        

        # The prediction starts, it is going to be predicted a given number of char
        for i in range(n_chars):

            # The numpy patterns is transformed into a tesor-type and reshaped
            pattern=pattern.type(torch.LongTensor)
            pattern=pattern.view(1,-1).to(device)

            # Make a prediction given the pattern
            prediction=model(pattern)
            
            
            prediction=softmax(prediction.type(torch.FloatTensor))
            

            # The prediction tensor is transformed into a numpy array
            prediction = prediction.squeeze().detach()
            
            # It is taken the idx with the highest probability
            arg_max = torch.argmax(prediction).unsqueeze(0).to(device)
            

            # The current pattern tensor is transformed into numpy array
            pattern = pattern.squeeze().detach()
            

            # The window is sliced 1 character to the right
            pattern = pattern[1:]
            # The new pattern is composed by the "old" pattern + the predicted character
            pattern = torch.cat((pattern, arg_max), dim=0)
            
            # The full prediction is saved
            full_prediction = torch.cat((full_prediction.to(device), arg_max), 0)
        
        full_prediction=full_prediction.cpu().detach().numpy()
        print("Prediction: \n")
        print(''.join([idx_to_char[value] for  value in full_prediction]), "\"")



if __name__ == '__main__':
	
	args = parameter_parser()
	
	# If you already have the trained weights
	if args.load_model == True:
		if os.path.exists(args.model):
			
			# Load and prepare sequences
			execution = Execution(args)
			execution.prepare_data()
			
			sequences = execution.sequences
			idx_to_char = execution.idx_to_char
			vocab_size = execution.vocab_size
			
			# Initialize the model
			model = TextGenerator(args, vocab_size).to(device)
			# Load weights
			model.load_state_dict(torch.load('C:\\Python\\Pytorch\\RNN and LSTM\\Text-Generation-BiLSTM-Pytorch\\weights\\textGenerator_model.pt'))
			
			# Text generator
			execution.generator(model, sequences, idx_to_char, 1000)

	
	# If you will train the model 		
	else:
		# Load and preprare the sequences
		execution = Execution(args)
		execution.prepare_data()
		
		# Training the model
		execution.train(args)

		sequences = execution.sequences
		idx_to_char = execution.idx_to_char
		vocab_size = execution.vocab_size
		
		# Initialize the model
		model = TextGenerator(args, vocab_size).to(device)
		# Load weights
		model.load_state_dict(torch.load('C:\\Python\\Pytorch\\RNN and LSTM\\Text-Generation-BiLSTM-Pytorch\\weights\\textGenerator_model.pt'))
		
		# Text generator
		execution.generator(model, sequences, idx_to_char, 1000)
        