# NLP

=============================================================================================
=============================================================================================

CS224n (Lecture and Exercise1-3 Done with Pytorch)

From: 

https://web.stanford.edu/class/cs224n/


A summary of a review

![CS224n NLP Course Structure(1)](https://user-images.githubusercontent.com/77312114/120138661-8b26c480-c209-11eb-9bc2-e8779b7448ad.png)


NLP_Lec: Lecture PowerPoint


NLP_Ass: Assignments

consulted

https://github.com/Luvata/CS224N-2019

=============================================================================================
=============================================================================================

Coursea NLP Specialization (All Done)

From:

https://www.coursera.org/specializations/natural-language-processing (sign the course for downloading models for transfer learning)

Consulted:

https://www.bilibili.com/video/BV1mA411v7Nv?p=33

https://github.com/amanjeetsahu/Natural-Language-Processing-Specialization

https://github.com/tsuirak/deeplearning.ai/tree/master/Natural%20Language%20Processing%20Specialization

Note: 

Course3 and Course4 are supported by a new deep learning platform called 'Trax', use Linux Ubuntu System for this. 

>>> C1_W1 Logistic Regression on sentiment analysis 

0, Preprocessing 

    from nltk.corpus import stopwords 
    from nltk.stem import PorterStemmer 
    from nltk.tokenize import TweetTokenizer 
    import re 
    import string 

    def process_tweet(tweet): 

    def build_freqs(tweets, ys): 
  
1, Logistic regression 

2, Extracting the features 

3, Training the model 

4, Test the logistic regression 

5, Error analysis 

\\\

>>> C1_W2 Naive bayes and sentiment anaysis 

0, Process the data 

1, Train the naive bayes model 

    def train_naive_bayes(freqs, train_x, train_y): 
        return logprior, loglikelihood
    
2, Test your naive bayes 

3, Filter words by ratio of positive to nagtive counts 

4, Error Analysis 

>>> C1_W3 Vector Space and Analogies

0, Predict the countries from Captitals 

    def cosine_similarity(A, B):

1, Ploting the vectors using PCA

    def compute_pca(X, n_components = 2):
        return X_reduced

\\\
        
>>> C1_W4 Naive Machine Translation and Local Sensitive Hashing  

0, Load the data 

1, Generate embedding and transform matrices 

    def get_matrices(en_fr, french_vecs, english_vecs): 
        return X, Y

    def compute_loss(X, Y, R):
        return loss 
        
    def computer_gradient(X, Y, R): 
        return gradient
        
2, Transformation matrix R 

    def align_embeddings(X, Y, train_steps, learning_rate): 
        return R 
        
3, The accuracy 

    def test_vocabulary(X, Y, R): 
        return accuracy
        
4, LSH and document search 

    def get_document_embedding(tweet, em_embeddings): 
    
    # store all docs into a dict 
    def get_document_vecs(all_docs, en_embeddings):

    # find the most similar tweets with LSH and similarity scores
    
>>> C2_W1 Auto Correction 

1, Data Preprocessing 

    import re 
    from collections import Counter 
    import numpy as np 
    import pandas as pd 
    
    # build a dict with lower case 
    def process_data(file_name):
        return words
        
    # get word count 
    def get_count(word_l): 
        return word_count_dict 
        
    # get word probability 
    def get_probs(word_count_dict): 
        return probs 
        
2, String Manipulations 

    def delete_letter() 
    
    def switch_letter() 
    
    def replace_letter() 
    
    def insert_letter() 
    
3, Combining the edits 

    def edit_one_letter() 
    
    ###
    
4, Minimun Edit Distence 

    def min_edit_distance(source, target, ins_cost = 1, del_cost = 1, rep_cost = 2): 
        return D, med
        
\\\ 
        
>>> C2_W2 POS Tagging and Hidden Markov Model 

0, Data preparation 

    # load in training corpus 
    # read in vocabulary, split 
    # assign unknown tokens 
    import string.punctuation 
    
1, POS Tagging

    def create_dictionaries(training_corpus, vocab):
        return emission_counts, transition_counts, tag_counts 
        
    # testing 
    def predict_pos(prep, y, emission_counts, vocab, states): 
        return accuracy 
        
2, Hidden Markov Models 

    def create_transition_matrix(alpha, tag_counts, transition_counts): 
        return A 
        
    def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
        return B 
        
    # get emission prob matrix 
    
3, Viterbi Algorithm 

\\\ 

>>> C2_W3 Language Modeling and N-gram 

1, Corpus preprocessing 

    # lower 
    # remove special char 
    # split 
    # tokenize 
    
    def sentence_to_trigram(tokenized_sentence):
    
2, Building the language model 

    def single_pass_trigram_count_matrix(corpus):
        return bigrams, vocabulary, count_matrix 
        
    # evaluation 
    def train_validation_test_split(data, train_percent, validation_percent):
        return train_data, validation_data, test_data
        
3, Out of vocabulary words 
    
    from collections import Counter
    vocabulary = Counter(word_counts).most_common(M)
    
    # smoothing 
    # back off

\\\

>>> C2_W4 Continusly Bag of Words and Neural Networks
    
1, CBOW model 

    # packages and helper functions 
    import nltk
    from nltk.tokenize import word_tokenize
    import numpy as np
    from collections import Counter
    from utils2 import sigmoid, get_batches, compute_pca, get_dict

    # frequency of the words 
    fdist = nltk.FreqDist(word for word in data)
    
    # index mapping 
    word2Ind, Ind2word = get_dict(data)
    
2, Training the model 

    def initialize_model(N, V, random_seed):
        return W1, W2, b1, b2 
        
    def softmax(z): 
        return yhat 
        
    def forward_prop(x, W1, W2, b1, b2): 
        return z, h
        
    def compute_cost(y, yhat, batch_size): 
        return cost 
        
    def backprop(x, yhat, y, h, W1, W2, b1, b2, batch_size): 
        return grad_W1, grad_W2, grad_b1, grad_b2
        
    def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):
        return W1, W2, b1, b2
        
3, Visualizing the word vectors 

\\\

>>> C3_W1 DNN and Sentiment Analysis 

1, Trax platform introduction 

    from trax import layers as tl 
    from trax import shapes 
    from trax import fastmath 
    
    # relu layer 
    relu = tl.Relu()
    
    # concatenate layer
    concat = tl.Concatenate() 
    
    # layer combinator 
    serial = tl.Serial(
        tl.LayerNorm(), 
        tl.Relu(), 
        times_two, 
        )
        
2, Class method recap 

3, Import the data 

    # load data
    # get length 
    # split data 
    
    # Set labels 
    train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)
    
    # def process_tweet(): tokenizing 
    
    # Build the vocabulary
    # include special tokens 
    # started with pad, end of line and unk tokens
    Vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2} 

    # Note that we build vocab using training data
    for tweet in train_x: 
        processed_tweet = process_tweet(tweet)
        for word in processed_tweet:
            if word not in Vocab: 
                Vocab[word] = len(Vocab)    

    # Convert a tweet to a tensor 
    def tweet_to_tnesor(): 
    
    # Create a batch generator 
    def data_generator(): 

4, Defining classes 

    # Relu class 
    class Relu(Layer): 
        return activation 
        
    # Dense class 
    class Dense(Layer): 
        return self.weights 
        
    # classifier function
    def classifier(vocab_size=len(Vocab), embedding_dim=256, output_dim=2, mode='train'):

        # Create embedding layer
        embed_layer = tl.Embedding(
            vocab_size=vocab_size, # Size of the vocabulary
            d_feature=embedding_dim)  # Embedding dimension

        # Create a mean layer, to create an "average" word embedding
        mean_layer = tl.Mean(axis=1)

        # Create a dense layer, one unit for each output
        dense_output_layer = tl.Dense(n_units =  output_dim)


        # Create the log softmax layer (no parameters needed)
        log_softmax_layer = tl.LogSoftmax()

        # Use tl.Serial to combine all layers
        # and create the classifier
        # of type trax.layers.combinators.Serial
        model = tl.Serial(
          embed_layer, # embedding layer
          mean_layer, # mean layer
          dense_output_layer, # dense output layer 
          log_softmax_layer # log softmax layer
        )
           
        # return the model of type
        return model
        
5, Training 

    from trax.supervised import training 
    
    # TrainTask 
    train_task = training.TrainTask(
    labeled_data=train_generator(batch_size=batch_size, shuffle=True),
    loss_layer=tl.CrossEntropyLoss(),
    optimizer=trax.optimizers.Adam(0.01),
    n_steps_per_checkpoint=10,
    ) 
    
    # EvalTask 
    eval_task = training.EvalTask(
    eval_task = training.EvalTask(
    labeled_data=val_generator(batch_size=batch_size, shuffle=True),
    metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
    )
    
    # Loop 
    def train_model(): 
    
6, Make prediction 
    
    # feed the tweet tensors into the model to get a prediction
    tmp_pred = training_loop.eval_model(tmp_inputs)
    
7, Evaluation and Test 

    def compute_accuracy(preds, y, y_weights):
        return accuracy, weighted_num_correct, sum_weights
        
    def test_model(): 
        return accuracy 
       
    def predict(sentence):
        return preds, sentiment
    
\\\

>>> C3_W2 GRU and Language Modeling  

1, Import the data 

    # remove leading and trailing whitespace
    pure_line = line.strip()
    
    # convert each line to tensor 
    def line_to_tensor()
    
    # batch generator 
    # mask = 1, pad = 0
    def data_generator(batch_size, max_length, data_lines, line_to_tensor, shuffle): 
    
    # repeat batch 
    import itertools 
    
2, Define the GRU model 

    def GRULM(vocab_size = 256, d_model = 512, n_layers = 2, mode = 'train'):
        model = tl.Serial(
          tl.ShiftRight(mode=mode), # Stack the ShiftRight layer
          tl.Embedding(vocab_size=vocab_size, d_feature=d_model), # Stack the embedding layer
          [tl.GRU(n_units=d_model) for _ in range(n_layers)], # Stack GRU layers of d_model 
          tl.Dense(n_units=vocab_size), # Dense layer
          tl.LogSoftmax()  # Log Softmax
        )
        return model       
    
3, Training 

    # training 
    def train_model(): 
        # data generator 
        # TrainTask 
        # EvalTask 
        # Loop 
        
\\\
>>> C3_W3 Named Entity Recongnition 

1, Exploring the data 

2, Importing the data 

    # vocab, tag_map 
    
    # data generator 
    def data_generator(
                        batch_size,
                        x, # list tensor
                        y, # list tensor 
                        shuffle, 
                        pad, 
                        verbose, 
                        )
        return X # size of (batch_size, max_len) 
               Y # size of (batch_size, max_len) 
               
3, Building the model 

    # LSTM model 
    def NER() 
        return model 
        
4, Train the model 

    # mask out the padding for training
    
    # def train_model(): 
        
    # use a pretrained model 
    
5, Evaluation 

    def evaluate_prediction(pred, labels, pad): 
    
    # prediction
    def predict(sentence, model, vocab, tag_map):
        return pred

        
