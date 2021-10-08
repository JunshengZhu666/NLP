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






