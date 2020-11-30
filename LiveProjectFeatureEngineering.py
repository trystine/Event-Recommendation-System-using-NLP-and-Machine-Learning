#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np


# In[2]:


data=pd.read_csv('LiveProject.csv', encoding='latin1')
data.head()


# In[3]:


data.groupby('Event')['Event'].count()


# In[4]:


data.groupby('Class')['Class'].count()


# In[5]:


#Text Cleaning and preparation
data['Input_1'] = data['Input'].str.replace("\r", " ")
data['Input_1'] = data['Input_1'].str.replace("\n", " ")
data['Input_1'] = data['Input_1'].str.replace("    ", " ")


# In[6]:


# " when quoting text
data['Input_1'] = data['Input_1'].str.replace('"', '')


# In[7]:


# Lowercasing the text
data['Input_2'] = data['Input_1'].str.lower()


# In[8]:


#Punctuation signs won't have any predicting power, so we'll just get rid of them.

punctuation_signs = list("?:!.,;")
data['Input_3'] = data['Input_2']

for punct_sign in punctuation_signs:
    data['Input_3'] = data['Input_3'].str.replace(punct_sign, '')


# In[9]:


#We'll also remove possessive pronoun terminations:

data['Input_4'] = data['Input_3'].str.replace("'s", "")


# In[10]:


# Downloading punkt and wordnet from NLTK
nltk.download('punkt')
print("------------------------------------------------------------")
nltk.download('wordnet')


# In[11]:


# Saving the lemmatizer into an object
wordnet_lemmatizer = WordNetLemmatizer()


# In[12]:


#In order to lemmatize, we have to iterate through every word:

nrows = len(data)
lemmatized_text_list = []

for row in range(0, nrows):
    
    # Create an empty list containing lemmatized words
    lemmatized_list = []
    
    # Save the text and its words into an object
    text = data.loc[row]['Input_4']
    text_words = text.split(" ")

    # Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    # Join the list
    lemmatized_text = " ".join(lemmatized_list)
    
    # Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)


# In[13]:


data['Input_5'] = lemmatized_text_list


# In[14]:


data.head()


# In[15]:


# Downloading the stop words list
nltk.download('stopwords')


# In[16]:


# Loading the stop words in english
stop_words = list(stopwords.words('english'))


# In[17]:


stop_words[0:10]


# In[18]:


data['Input_6'] = data['Input_5']

for stop_word in stop_words:

    regex_stopword = r"\b" + stop_word + r"\b"
    data['Input_6'] = data['Input_6'].str.replace(regex_stopword, '')


# In[19]:


data.head()


# In[20]:


#delete intermediate columns
list_columns = ["Input", "Class", "Event", "Input_6"]
data = data[list_columns]

data = data.rename(columns={'Input_6': 'Input_Parsed'})


# In[21]:


data.head()


# In[22]:


#Now we will start with label coding
class_codes = {
    'Artificial Intelligence': 0,
    'Web Development': 1,
    'Cloud Computing': 2,
    'Cyber Security': 3,
    'Digital Marketing': 4,
    'IOT': 5,
    'Other': 6,
    'Not defined': 7
}

event_codes={
    'Workshops': 0,
    'Courses': 1,
    'Jobs': 2,
    'Webinars': 3,
    'Hackathons':4,
}


# In[23]:


# Category mapping
data['Class_Code'] = data['Class']
data['Event_Code'] = data['Event']
data = data.replace({'Class_Code':class_codes})
data = data.replace({'Event_Code':event_codes})


# In[24]:


data.head()


# In[25]:


#Train - test split
#We'll set apart a test set to prove the quality of our models. We'll do Cross Validation in the train set in order to tune the hyperparameters and then test performance on the unseen data of the test set.

X_train, X_test, y_train, y_test = train_test_split(data['Input_Parsed'], 
                                                    data['Class_Code'],
                                                    test_size=0.30, 
                                                    random_state=8)


# In[26]:


#We'll use TF-IDF Vectors as features.

#We have to define the different parameters:

#ngram_range: We want to consider both unigrams and bigrams.
#max_df: When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold
#min_df: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
#max_features: If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
#See TfidfVectorizer? for further detail.

#It needs to be mentioned that we are implicitly scaling our data when representing it as TF-IDF features with the argument norm.

# Parameter election
ngram_range = (1,2)
min_df = 1 
max_df = 10. 
max_features = 300


# In[27]:


tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)


# In[28]:


#Please note that we have fitted and then transformed the training set, but we have only transformed the test set.

#We can use the Chi squared test in order to see what unigrams and bigrams are most correlated with each category:

from sklearn.feature_selection import chi2
import numpy as np

for Product, category_id in sorted(class_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")


# In[30]:


#As we can see, the unigrams correspond well to their category. However, bigrams do not. If we get the bigrams in our features:

bigrams


# In[31]:


#We can see there are only six. This means the unigrams have more correlation with the category than the bigrams, and since we're restricting the number of features to the most representative 300, only a few bigrams are being considered.

#Let's save the files we'll need in the next steps:

# X_train
with open('Pickles/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)
    
# X_test    
with open('Pickles/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)
    
# y_train
with open('Pickles/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)
    
# y_test
with open('Pickles/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)
    
# df
with open('Pickles/df.pickle', 'wb') as output:
    pickle.dump(data, output)
    
# features_train
with open('Pickles/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('Pickles/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('Pickles/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('Pickles/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)
    
# TF-IDF object
with open('Pickles/tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)


# In[32]:


#Train - test split for second classifier
#We'll set apart a test set to prove the quality of our models. We'll do Cross Validation in the train set in order to tune the hyperparameters and then test performance on the unseen data of the test set.

X_train, X_test, y_train, y_test = train_test_split(data['Input_Parsed'], 
                                                    data['Event_Code'],
                                                    test_size=0.30, 
                                                    random_state=8)


# In[33]:


ngram_range = (1,2)
min_df = 1 
max_df = 10. 
max_features = 300


# In[34]:


tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)


# In[35]:


from sklearn.feature_selection import chi2
import numpy as np

for Product, category_id in sorted(event_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")


# In[36]:


#As we can see, the unigrams correspond well to their category. However, bigrams do not. If we get the bigrams in our features:

bigrams


# In[37]:


#Let's save the files we'll need in the next steps:

# X_train2
with open('Pickles/X_train2.pickle', 'wb') as output:
    pickle.dump(X_train, output)
    
# X_test    
with open('Pickles/X_test2.pickle', 'wb') as output:
    pickle.dump(X_test, output)
    
# y_train
with open('Pickles/y_train2.pickle', 'wb') as output:
    pickle.dump(y_train, output)
    
# y_test
with open('Pickles/y_test2.pickle', 'wb') as output:
    pickle.dump(y_test, output)
    
# df
with open('Pickles/df.pickle', 'wb') as output:
    pickle.dump(data, output)
    
# features_train
with open('Pickles/features_train2.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('Pickles/labels_train2.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('Pickles/features_test2.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('Pickles/labels_test2.pickle', 'wb') as output:
    pickle.dump(labels_test, output)
    
# TF-IDF object
with open('Pickles/tfidf2.pickle', 'wb') as output:
    pickle.dump(tfidf, output)


# In[38]:


data.groupby('Event_Code')['Event_Code'].count()


# In[39]:


data.groupby('Class_Code')['Class_Code'].count()


# In[ ]:




