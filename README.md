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

\

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



    
    


