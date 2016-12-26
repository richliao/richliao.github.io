---
layout: post
comments: true
title: "Text Classification, Part I - convolutional"
date: "2016-11-26 16:35:30 -0500"
categories: jekyll disqus
---

Text classification is a very classical problem. The goal is to classify documents into a fixed number of predefined categories, given a variable length of text bodies. It is widely use in sentimental analysis (IMDB, YELP reviews classification), stock market sentimental analysis, to GOOGLE's smart email reply. This is a very active research area both in academia and industry. In the following series of posts, I will try to present a few different approaches and compare their performances. Ultimately, the goal for me is to implement the paper [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf).
Given the limitation of data set I have, all exercises are based on Kaggle's IMDB dataset. And implementation are all based on Keras.

## Text classification using CNN ##

In this first post, I will look into how to use convolutional neural network to build a classifier, particularly [Convolutional Neural Networks for Sentence Classification - Yoo Kim](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf).

First use BeautifulSoup to remove some html tags and remove some unwanted characters.

{% highlight python %}
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()

texts = []
labels = []

for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx])
    texts.append(clean_str(text.get_text().encode('ascii','ignore')))
    labels.append(data_train.sentiment[idx])

{% endhighlight %}

Keras has provide very nice text processing functions.

{% highlight python %}
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

{% endhighlight %}

For this project, I have used [Google Glove 6B vector 100d](http://nlp.stanford.edu/projects/glove/). For Unknown word, the following code will just randomize its vector.

{% highlight python %}
GLOVE_DIR = "~/data/glove"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
{% endhighlight %}


### A simplified Convolutional ###
First, I will just use a very simple convolutional architecture here. Simply use total 128 filters with size 5 and max pooling of 5 and 35, following the sample from [this blog](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)

{% highlight python %}
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(5)(l_cov2)
l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
l_flat = Flatten()(l_pool3)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')
{% endhighlight %}

The accuracy we can achieve is **89%**

### Complex Convolutional neural network ###

In Yoon Kim's paper, multiple filters have been applied. This can be easily implemented using Keras Merge Layer.

{% include image.html url="/images/YoonKim_ConvtextClassifier.png" description="Convolutional network with multiple filter sizes" %}

{% highlight python %}
convs = []
filter_sizes = [3,4,5]

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
    l_conv = Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convs.append(l_pool)

l_merge = Merge(mode='concat', concat_axis=1)(convs)
l_cov1= Conv1D(128, 5, activation='relu')(l_merge)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(30)(l_cov2)
l_flat = Flatten()(l_pool2)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)
{% endhighlight %}

As you can see, the result slighly improved to **90%**

To achieve the best performances, we can
1) fine tune hyper parameters
2) improve text preprocessing.

## Conclusion ##
Based on the observation, the complexity of convolutional neural network doesn't play much of the role of improving performance, at least using this small dataset. We might be able to see significant performance gap with larger dataset available, which I won't be able to verify here. One observation I have is allowing the embedding layer training or not does improve the performance, also using pretrained Google Glove word vectors. In both cases, I can see performance improved from 82% to close to 90%.
