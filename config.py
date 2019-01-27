class DefaultConfigs(object):
    path = '/home/gaurav/Downloads/Personal Emotional Doctor/'
    fasttext_encoding_path = '/home/gaurav/Downloads/wiki.en/wiki.en.bin'
    gru_layers = 2
    gru_hidden_units = 256
    lr = 3e-3
    # Currently not making batches, can make use of parallelization using approach similar to language modeling
    batch_size = 1
    epochs = 50


config = DefaultConfigs()

