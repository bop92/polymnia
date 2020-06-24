import torch
import time
import numpy

class model(torch.nn.Module):

    
    #TODO simplify arg passing
    def __init__(self, embedding_dims, vocabulary_size, weight_dist, batch_size, window_size, negative_samples, device):
        super(model, self).__init__()
        self.device=device
        self.weight_dist = weight_dist
        self.batch_size = batch_size
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.vocabulary_size = vocabulary_size
        

        self.i_vec = torch.nn.Embedding(vocabulary_size, embedding_dims, padding_idx=0)
        self.o_vec = torch.nn.Embedding(vocabulary_size, embedding_dims, padding_idx=0)

        self.i_vec.weight = torch.nn.Parameter(
                torch.cat([torch.zeros(1, embedding_dims), 
                torch.FloatTensor(vocabulary_size - 1, embedding_dims)
                .uniform_(-0.5 / embedding_dims, 0.5 / embedding_dims)]))

        self.o_vec.weight = torch.nn.Parameter(
                torch.cat([torch.zeros(1, embedding_dims), 
                torch.FloatTensor(vocabulary_size - 1, embedding_dims)
                .uniform_(-0.5 / embedding_dims, 0.5 / embedding_dims)]))
               
        #We can perhaps convert dtype to 16
        #this will only work on cuda it seems ...
        '''
        self.i_vec.weight = torch.nn.Parameter(
                torch.cat([torch.zeros(1, embedding_dims), 
                torch.HalfTensor(vocabulary_size - 1, embedding_dims)]))

        self.o_vec.weight = torch.nn.Parameter(
                torch.cat([torch.zeros(1, embedding_dims), 
                torch.HalfTensor(vocabulary_size - 1, embedding_dims)]))
        '''      

        self.i_vec.weight.requires_grad = True
        self.o_vec.weight.requires_grad = True


    def forward(self, i_word, o_word):
        n_word = numpy.random.choice(self.vocabulary_size, size=(self.batch_size * self.window_size * 2 * self.negative_samples), p=self.weight_dist)
        n_word = torch.LongTensor(n_word).view(self.batch_size, -1)
        print(n_word.shape)
        i_vectors = self.i_vec(torch.LongTensor(i_word).to(self.device)).unsqueeze(2)
        o_vectors = self.o_vec(torch.LongTensor(o_word).to(self.device))
        n_vectors = self.o_vec(n_word.to(self.device)).neg()

        oloss = torch.bmm(o_vectors, i_vectors).squeeze().sigmoid().log().mean(1)
        nloss = torch.bmm(n_vectors, i_vectors).squeeze().sigmoid().log().view(-1, self.window_size * 2, self.negative_samples).sum(2).mean(1)
        print(oloss)
        print(nloss)
        return -(oloss + nloss).mean()

