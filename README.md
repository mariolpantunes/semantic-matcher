# semantic_search

1. Create a file secret.py and inside place the Usearch API key <code>key='...'</code>

2. Download the [dataset](https://www.kaggle.com/datasets/mantunes/corpus-for-semantic-matching) and [scenario](https://www.kaggle.com/datasets/mantunes/semantic-service-discovery-in-ndn), extract the files and place them in the repository folder.

3. Download pretrained fasttext https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
    - put it in `results/fasttext/pretrained/` and `results/fasttext/pretrained_optimized/`
    - rename it `pretrained.vec`

4. Download and extract the pretrained glove https://nlp.stanford.edu/data/glove.6B.zip
    - put it in `results/glove/pretrained_{vector_size}/`
5. Install python libraries
    - `pip install -r requirements.txt`
6. Install the nltk stopwords, punkt in the python env
<code>
import nltk
nltk.download('stopwords')
nltk.download('punkt')
</code>

7. Download and install [GloVe](https://nlp.stanford.edu/projects/glove/)
    -  Please follow the instructions on the official website and move the executables to a glove folder in this repository.


## Authors

* **Mário Antunes** - [mariolpantunes](https://github.com/mariolpantunes)
* **Rafael Teixeira** - [rgtzths](https://github.com/rgtzths)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
