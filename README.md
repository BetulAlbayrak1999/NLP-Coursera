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
