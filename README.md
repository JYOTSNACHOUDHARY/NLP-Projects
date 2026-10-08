# 📚 NLP Projects

**Hands-on natural language processing, from classical linear-algebra methods to neural text classifiers, with a focus on *why* each technique works.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-text%20processing-154F5B)
![NumPy](https://img.shields.io/badge/NumPy%20%2F%20SciPy-013243?logo=numpy&logoColor=white)

---

## Projects at a glance

| Project | Problem | Approach | Result |
|---|---|---|---|
| [**Fake vs. real news**](Fake_and_Real_news/classification-with-custom-embeddings.ipynb) | Binary classification of news articles | Engineered class-exclusive vocabulary → trained embeddings → Conv1D network | **98.25% test accuracy** |
| [**Text similarity**](Text_Similarity.ipynb) | How semantically close are two texts? | GloVe sentence vectors with cosine similarity · Smooth Inverse Frequency · Latent Semantic Indexing (SVD) | Matches paraphrases with no shared words; ranks documents by query |
| [**IMDB sentiment**](IMDB%20movie%20review/sentiment_analysis.ipynb) | Positive or negative movie reviews | Cleaning pipeline → bag-of-words baseline → Keras model with trained embeddings | Work in progress |
| [**Markov chains**](Markov_Chain.py) | Stationary distribution of a stochastic process | Monte Carlo simulation, repeated matrix powers, and left eigenvectors | All three agree: π ≈ [0.352, 0.211, 0.437] |

---

## 🔍 Fake vs. real news: classification with custom embeddings

**Insight:** the most frequent words in fake news that *never* appear among the most frequent words in real news (and the other way round) carry most of the signal. Training on that **selective vocabulary**, instead of the whole corpus vocabulary, gives a smaller and more discriminative input space.

**Pipeline**
1. **Cleaning:** strip non-ASCII characters, URLs and Twitter handles, and remove stop words. Punctuation is deliberately kept as an expressive signal.
2. **Exploratory analysis:** treemaps of class-exclusive word frequencies and word clouds for each class
3. **Feature engineering:** find the words exclusive to each class using only the training split (avoiding leakage)
4. **Model:** Keras tokenizer over the selective vocabulary → 200-dim trainable `Embedding` → 2 × (`Conv1D` + max-pool) → dense sigmoid classifier
5. **Evaluation:** a held-out 20% test split

```text
Train (epoch 5): loss 0.0159 · accuracy 99.46%
Test           : loss 0.0740 · accuracy 98.25%
```

## 📐 Text similarity: three generations of techniques

Takes *"President greets the press in Chicago"* and *"Obama speaks to media Illinois"*, two sentences that share **no** words, and measures how similar they are:

- **GloVe averaging and cosine similarity:** tokenise, remove stop words, average the word vectors, compare. Includes a **word-to-word similarity heatmap**.
- **Smooth Inverse Frequency (SIF):** why a frequency-weighted average beats a plain mean for sentence embeddings.
- **Latent Semantic Indexing:** builds a term-document matrix, applies **SVD**, keeps a rank-2 approximation, projects the query into concept space, and ranks documents by cosine similarity.

## 🎬 IMDB sentiment analysis

An end-to-end text-classification pipeline on IMDB movie reviews: removing HTML with BeautifulSoup, a data-driven decision on whether stop words matter, punctuation stripping, tokenisation, a `CountVectorizer` baseline, and a Keras model with trained 100-dim embeddings over a 10k-word vocabulary. *Training is still in progress.*

## 🎲 Markov chains: three ways to find equilibrium

Using a three-state food-choice chain (Burger, Pizza, Hot dog), the notebook:
- Simulates random walks over the transition matrix
- Finds the **stationary distribution** three independent ways (a Monte Carlo run of 10⁶ steps, matrix powers, and the left eigenvector for eigenvalue 1) and shows that they converge
- Calculates the probability of a given state sequence, such as *P(Pizza → Hot dog → Hot dog → Burger)* ≈ 0.037

This is the same mathematics behind n-gram language models and PageRank.

---

## Running the notebooks

```bash
git clone https://github.com/JYOTSNACHOUDHARY/NLP-Projects.git
cd NLP-Projects
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas scikit-learn nltk tensorflow beautifulsoup4 matplotlib wordcloud squarify jupyter
python -m nltk.downloader stopwords punkt
jupyter lab
```

**Datasets** (not included in the repo):
- [Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle)
- [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle), saved as `IMDB Dataset.csv`
- [GloVe pre-trained word vectors](https://nlp.stanford.edu/projects/glove/) (Stanford NLP); set `embedding_path` in the text-similarity notebook

---

<p align="center">Built by <a href="https://github.com/JYOTSNACHOUDHARY">Jyotsna Choudhary</a> · <a href="https://www.linkedin.com/in/jyotsna-c/">LinkedIn</a> · <a href="https://www.youtube.com/@LearnHiddenLayers">YouTube</a></p>
