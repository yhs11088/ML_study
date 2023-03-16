import numpy as np
import torch
import torch.nn as nn

class CPC(nn.Module):
    def __init__(self, timestep, batch_size, seq_len):

        super(CPC, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 512, kernel_size = 10, stride = 5, padding = 3, bias = False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 8, stride = 4, padding = 2, bias = False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True)
        )
        # GRU
        self.gru = nn.GRU(input_size = 512, hidden_size = 256, num_layers = 1, bidirectional = False, batch_first = True)
        # Linear layers
        self.Wk = nn.ModuleList([nn.Linear(in_features = 256, out_features = 512) for i in range(timestep)])
        '''
        self.softmax = nn.Softmax()            # Softmax (default : along the last dim)
        self.lsoftmax = nn.LogSoftmax()        # LogSoftmax (default : along the last dim)
        '''
        self.softmax = nn.Softmax(dim = 0)     # Softmax along 0-th dim (batch of positive & negative samples)
        self.lsoftmax = nn.LogSoftmax(dim = 0) # LogSoftmax along 0-th dim (batch of positive & negative samples)

    def forward(self, X, hidden):

        # shape of X = (batch, audio_channel, audio_length) = (8, 1, 20480)
        # shape of hidden = (1, batch, output_dim) = (1, 8, 256)
        batch = X.size()[0]

        # 1. pick random time stamp to make prediction & calculate loss
        '''
        t_samples = torch.randint(self.seq_len/160-self.timestep, size = (1,)).long() # shape = (1,)
        '''
        t_samples = torch.randint(self.seq_len//160-self.timestep, size = (1,)).long() # shape = (1,)

        # 2. Encoder
        Z = self.encoder(X) # shape = (batch, hidden_dim, repr_length) = (8, 512, 128)

        # 3. Reshape Z
        Z = Z.transpose(1, 2) # shape = (batch, repr_length, hidden_dim) = (8, 128, 512)

        # 4. Pick answers (i.e. true k-th future Z from t_samples)
        hidden_dim = Z.size()[-1]
        encode_samples = torch.empty((self.timestep, batch, hidden_dim)).float() # shape = (12, 8, 512)
        for k in range(1, self.timestep+1):
            encode_samples[k-1] = Z[:,t_samples+k,:].view(batch, hidden_dim)

        # 5. GRU
        forward_seq = Z[:,:t_samples+1,:] # past & current Z (= GRU input) (shape = (batch, t_samples+1, hidden_dim) = (8, t_samples+1, 512))
        output, hidden = self.gru(forward_seq, hidden) # shape of output = (batch, t_samples+1, output_dim) = (8, t_samples+1, 256)

        # 6. Linear 
        output_dim = output.size()[-1]
        c_t = output[:,t_samples,:].view(batch, output_dim)    # last C (shape = (batch, output_dim) = (8, 256))
        pred = torch.empty((self.timestep, batch, hidden_dim)) # model predictions for k-th future Z
        for k in range(self.timestep):
            pred[k] = self.Wk[k](c_t)
        
        # 7. InfoNCE loss
        nce = 0

        for k in range(self.timestep):

            # 7-1. Dot product between true & predicted k-th future Z's
            # --> total[i,j] = dot product between encode_samples[k,i] and pred[k,j]
            #                = (if i == j) dot product with positive sample
            #                  (otherwise) dot product with negative sample
            total = torch.mm(encode_samples[k], torch.transpose(pred[k], 0, 1)) # shape = (batch, batch) = (8, 8)

            # 7-2. Calculate accuracy
            # --> If model is prefect,
            #     torch.argmax(self.softmax(total), dim = 0) = [0, 1, ..., batch-1]
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim = 0), torch.arange(batch)))

            # 7-3. Calculate InfoNCE loss
            # --> We need LogSoftmax of positive sample only
            # --> Thus, we only need torch.diag(self.lsoftmax(total)) & calculate its sum
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        
        nce /= (-1. * batch * self.timestep)    # average InfoNCE loss
        accuracy = 1. * correct.item() / batch  # accuracy of {self.timestep}-th future prediction

        return accuracy, nce, hidden
        #return accuracy, nce, hidden, Z, t_samples         # ***** When comparing with Spijkervet

    def predict(self, X, hidden):

        # shape of X = (batch, audio_channel, audio_length) = (8, 1, 20480)
        batch = X.size()[0]

        # 1. Encoder
        Z = self.encoder(X) # shape = (batch, hidden_dim, repr_length) = (8, 512, 128)

        # 2. Reshape Z
        Z = Z.transpose(1, 2) # shape = (batch, repr_length, hidden_dim) = (8, 128, 512)

        # 3. GRU
        output, hidden = self.gru(Z, hidden) # shape of output = (batch, repr_length, output_dim) = (8, 128, 256)

        return output, hidden