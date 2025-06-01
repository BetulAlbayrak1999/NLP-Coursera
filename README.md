# NLP-Coursera

### Preprocessing
In this lab, we will be exploring how to preprocess tweets for sentiment analysis. We will provide a function for preprocessing tweets during this week's assignment, but it is still good to know what is going on under the hood. By the end of this lecture, you will see how to use the NLTK package to perform a preprocessing pipeline for Twitter datasets.

### Word Frequencies
Building and Visualizing Word Frequencies
In this lab, we will focus on the build_freqs() helper function and visualizing a dataset fed into it. In our goal of tweet sentiment analysis, this function will build a dictionary where we can look up how many times a word appears in the lists of positive or negative tweets. This will be very helpful when extracting the features of the dataset in the week's programming assignment. Let's see how this function is implemented under the hood in this notebook.

### Utils
The **Utils** notebook provides essential helper functions for preprocessing and feature engineering in tweet sentiment analysis. The `process_tweet` function cleans raw tweets by removing stock tickers, retweet indicators, hyperlinks, and hashtags, then tokenizes the text, removes stopwords and punctuation, and applies stemming to normalize the words. This ensures that tweets are reduced to meaningful tokens suitable for machine learning. The `build_freqs` function constructs a frequency dictionary that maps each pair $(\text{word}, y)$, where *word* is a token from a processed tweet and *y* is its sentiment label (0 for negative, 1 for positive), to its frequency in the dataset. These utilities form the foundation for feature extraction and model training.

### Logistic_Regression_Decide_Sentiment
The **Predict Tweets** notebook implements a sentiment analysis system using `logistic regression` to classify tweets as positive or negative. It begins by preprocessing tweets through cleaning (removing URLs, mentions, hashtags, etc.), `tokenizing`, removing `stopwords` and `punctuation`, and applying `stemming`. A frequency dictionary is then built to map each word-sentiment pair $(\text{word}, y)$ to its occurrence count. Each tweet is transformed into a feature vector $\mathbf{x} = [1, \text{positive\_count}, \text{negative\_count}]$, which is used to train a logistic regression model with `gradient descent`. The model learns a weight vector $\theta$, and predictions are made using the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$, where $z = \mathbf{x} \cdot \theta$. The notebook concludes by evaluating the model’s accuracy and demonstrating its predictions on example tweets.

### Naive Bayes
This notebook implements a Naive Bayes classifier to detect sentiment `(positive or negative)` in tweets. It starts by preprocessing tweets (e.g., removing stopwords and symbols), then counts how frequently each word appears in positive and negative tweets. Using these frequencies, it calculates the log prior (overall sentiment bias) and `log likelihoods` (how much each word influences sentiment). The model then predicts the sentiment of new tweets based on the presence of these words. Finally, it evaluates accuracy and extracts the most influential positive or negative words using a ratio-based threshold.

### manipulating_word_embeddings
This Jupyter Notebook, titled `6manipulating_word_embeddings.ipynb`, is part of the NLP-Coursera repository by Betul Albayrak. It focuses on the practical aspects of manipulating word embeddings, which are crucial for various natural language processing tasks. The notebook provides hands-on examples and code snippets that demonstrate how to load, visualize, and modify word embeddings, enabling users to understand their properties and applications in machine learning models. This resource is ideal for learners looking to deepen their understanding of word embeddings in NLP.
