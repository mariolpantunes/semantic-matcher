cache='dataset'
configs = [
    #pretrained models
    #{
    #    "model":'glove',
    #    "pretrained":"pretrained",
    #    "vector_size":50
    #},
    #{
    #    "model":'glove',
    #    "pretrained":"pretrained",
    #    "vector_size":100
    #},
    #{
    #    "model":'glove',
    #    "pretrained":"pretrained",
    #    "vector_size":200
    #},
    #{
    #    "model":'glove',
    #    "pretrained":"pretrained",
    #    "vector_size":300
    #},
    #{
    #    "model":'fasttext',
    #    "pretrained":"pretrained",
    #},
    #{
    #    "model":'word2vec',
    #    "pretrained":"pretrained",
    #},
    #{
    #    "model":'sbert',
    #    "pretrained":"pretrained",
    #    "semantic_training":False
    #},
    #baselines
    #{
    #    "model":'string',
    #    "st":0
    #},
    #{
    #    "model":'levenshtein',
    #    "st":2
    #},
    #Optimized
    #{
    #    "model":'sbert',
    #    "pretrained":"pretrained_optimized",
    #    "semantic_training":False
    #},
    {
        "model":'sbert',
        "pretrained":"pretrained_optimized",
        "semantic_training":True
    },
    {
        "model":'sbert',
        "pretrained":"pretrained",
        "semantic_training":True
    },
    {
        "model":'fasttext',
        "pretrained":"pretrained_optimized",
        "n_threads":32
    },

    #trained_from_scratch
    {
        "model":'glove',
        "window_size":3,
        "vector_size":43,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'glove',
        "window_size":3,
        "vector_size":50,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'glove',
        "window_size":5,
        "vector_size":78,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'glove',
        "window_size":5,
        "vector_size":88,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'glove',
        "window_size":7,
        "vector_size":108,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'glove',
        "window_size":7,
        "vector_size":120,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'fasttext',
        "window_size":3,
        "vector_size":43,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'fasttext',
        "window_size":3,
        "vector_size":50,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'fasttext',
        "window_size":5,
        "vector_size":78,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'fasttext',
        "window_size":5,
        "vector_size":88,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'fasttext',
        "window_size":7,
        "vector_size":108,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'fasttext',
        "window_size":7,
        "vector_size":120,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'word2vec',
        "window_size":3,
        "vector_size":43,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'word2vec',
        "window_size":3,
        "vector_size":50,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'word2vec',
        "window_size":5,
        "vector_size":78,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'word2vec',
        "window_size":5,
        "vector_size":88,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'word2vec',
        "window_size":7,
        "vector_size":108,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'word2vec',
        "window_size":7,
        "vector_size":120,
        "pretrained":"from_scratch",
        "n_threads":32
    },
    {
        "model":'sbert',
        "vector_size":36,
        "pretrained":"from_scratch",
        "semantic_training":True
    },
    {
        "model":'sbert',
        "vector_size":48,
        "pretrained":"from_scratch",
        "semantic_training":True
    },
    {
        "model":'sbert',
        "vector_size":72,
        "pretrained":"from_scratch",
        "semantic_training":True
    },
    {
        "model":'sbert',
        "vector_size":84,
        "pretrained":"from_scratch",
        "semantic_training":True
    },
    {
        "model":'sbert',
        "vector_size":108,
        "pretrained":"from_scratch",
        "semantic_training":True
    },
    {
        "model":'sbert',
        "vector_size":120,
        "pretrained":"from_scratch",
        "semantic_training":True
    },
]
