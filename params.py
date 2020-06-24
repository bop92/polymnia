# the window size for generating the skip grams (non inclusive)
window_size = 2
# embedding dimensions (sometimes referred to as feature size)
embedding_dims = 100
# the learning rate
learning_rate = .01
# the batch size. this is currently measured by the number of skip grams loaded
batch_size = 20000
# the number of negative samples for calculating loss
negative_samples = 25
# the constant used for the unknown character 
unk = "<UNK>"
