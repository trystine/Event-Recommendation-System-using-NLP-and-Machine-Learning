#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import punkt
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


#trained models
path_models = "Models/"

# First Classifier
path_rfc = path_models + 'best_rfc.pickle'
with open(path_rfc, 'rb') as data:
    rfc_model = pickle.load(data)
    
#Second Classifier
path_rfc2 = path_models + 'best_rfc2.pickle'
with open(path_rfc2, 'rb') as data:
    rfc_model2 = pickle.load(data)

#loading employee data
path_df = "Pickles/employee_list.pickle"
with open(path_df, 'rb') as output:
    emp = pickle.load(output)


# In[3]:


#TF-IDF object
path_tfidf = "Pickles/tfidf.pickle"
with open(path_tfidf, 'rb') as data:
    tfidf = pickle.load(data)
    
#Another Classifier
path_tfidf2 = "Pickles/tfidf2.pickle"
with open(path_tfidf2, 'rb') as data:
    tfidf2 = pickle.load(data)


# In[4]:


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


# In[5]:


#feature engineering
punctuation_signs = list("?:!.,;")
stop_words = list(stopwords.words('english'))

def create_features_from_text(text):
    
    # Dataframe creation
    lemmatized_text_list = []
    df = pd.DataFrame(columns=['Input'])
    df.loc[0] = text
    df['Input_1'] = df['Input'].str.replace("\r", " ")
    df['Input_1'] = df['Input_1'].str.replace("\n", " ")
    df['Input_1'] = df['Input_1'].str.replace("    ", " ")
    df['Input_1'] = df['Input_1'].str.replace('"', '')
    df['Input_2'] = df['Input_1'].str.lower()
    df['Input_3'] = df['Input_2']
    for punct_sign in punctuation_signs:
        df['Input_3'] = df['Input_3'].str.replace(punct_sign, '')
    df['Input_4'] = df['Input_3'].str.replace("'s", "")
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    text = df.loc[0]['Input_4']
    text_words = text.split(" ")
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    lemmatized_text = " ".join(lemmatized_list)    
    lemmatized_text_list.append(lemmatized_text)
    df['Input_5'] = lemmatized_text_list
    df['Input_6'] = df['Input_5']
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Input_6'] = df['Input_6'].str.replace(regex_stopword, '')
    df = df['Input_6']
    df = df.rename(columns={'Input_6': 'Input_Parsed'})
    
    # TF-IDF
    features = tfidf.transform(df).toarray()
    
    return features


# In[6]:


#Now let's write a function that tells us the category given the category code:
def get_category_name(category_id):
    for category, id_ in class_codes.items():    
        if id_ == category_id:
            return category

def get_category_name2(category_id):
    for category, id_ in event_codes.items():    
        if id_ == category_id:
            return category


# In[7]:


#Finally, let's write a function that includes the whole process:
def predict_from_text(text):
    
    # Predict using the input model
    prediction_rfc = rfc_model.predict(create_features_from_text(text))[0]
    prediction_rfc_proba = rfc_model.predict_proba(create_features_from_text(text))[0]
    
    # Return result
    category_rfc = get_category_name(prediction_rfc)
    return category_rfc
    #print("The predicted category using the RFC model is %s." %(category_rfc) )
    #print("The conditional probability is: %a" %(prediction_rfc_proba.max()*100))

#For second process
def predict_from_text2(text):
    
    # Predict using the input model
    prediction_rfc = rfc_model2.predict(create_features_from_text(text))[0]
    prediction_rfc_proba = rfc_model2.predict_proba(create_features_from_text(text))[0]
    
    # Return result
    category_rfc = get_category_name2(prediction_rfc)
    return category_rfc


# In[8]:


#Reading Input file
input_text=pd.read_csv('input.csv')
input_text


# In[9]:


text1= input_text.iloc[0]
text2=input_text.iloc[1]
text3=input_text.iloc[2]
text4=input_text.iloc[3]
#text4=pd.DataFrame(text4)
#text4.reset_index(drop=True, inplace=True)
#text4=str(text4)
#text1=pd.DataFrame(text1)
#text1=str(text1)
print(text1)
print(text2)
print(text3)
print(text4)


# In[10]:


#Hackathon word added as the classifier is not smart enough to classify codeathon as hackathon. Reason being the training
#data collected is not large. A proper sampled data is required which may require financial assistance to buy from vendors
#Since this project is made in short period of time we are doing modifications in input as well
#Do need we have only consider few domains and hence coding as a domain is eliminated. Hence classifier predicts text 3 
#domain as not defined


# In[11]:


text1_domain=predict_from_text(text1)
text1_event_type=predict_from_text2(text1)
text2_domain=predict_from_text(text2)
text2_event_type=predict_from_text2(text2)
text3_domain=predict_from_text(text3)
text3_event_type=predict_from_text2(text3)
text4_domain=predict_from_text(text4)
text4_event_type=predict_from_text2(text4)


# In[12]:


print("Event1 output",':',"Domain-",text1_domain,",","Event Type-", text1_event_type)
print("Event2 output",':',"Domain-",text2_domain,",","Event Type-", text2_event_type)
print("Event3 output",':',"Domain-",text3_domain,",","Event Type-", text3_event_type)
print("Event4 output",':',"Domain-",text4_domain,",","Event Type-", text4_event_type)


# In[13]:


#Now let us see the employees that are interested
#We are going to display the names of employees with matching domain and event type based on various inputs


# In[61]:


Input1_output=pd.DataFrame(emp[(emp['Domain']==text1_domain) & ((emp['Event1']==text1_event_type) | (emp['Event2']==text1_event_type))])
Input2_output=emp[(emp['Domain']==text2_domain) & ((emp['Event1']==text2_event_type) | (emp['Event2']==text2_event_type))]
Input3_output=emp[(emp['Domain']==text3_domain) & ((emp['Event1']==text3_event_type) | (emp['Event2']==text3_event_type))]
Input4_output=emp[(emp['Domain']==text4_domain) & ((emp['Event1']==text4_event_type) | (emp['Event2']==text4_event_type))]
print("Employee's Interested for Input1:")
Input1_output = Input1_output.set_index('Name')
Input1_output


# In[62]:


print("Employee's Interested for Input2:")
Input2_output = Input2_output.set_index('Name')
Input2_output


# In[63]:


#No one so we will write No one as a name input
Input2_output = Input2_output.append({'Name' : 'No employee for text2'},ignore_index=True)
Input2_output = Input2_output.set_index('Name')
Input2_output


# In[64]:


print("Employee's Interested for Input3:")
Input3_output = Input3_output.set_index('Name')
Input3_output


# In[65]:


#No one so we will write No one as a name input
Input3_output = Input3_output.append({'Name' : 'No employee for text3'},ignore_index=True)
Input3_output = Input3_output.set_index('Name')
Input3_output


# In[66]:


print("Employee's Interested for Input4:")
Input4_output = Input4_output.set_index('Name')
Input4_output


# In[67]:


frames = [Input1_output,Input2_output,Input3_output,Input4_output]
Employees_list=pd.concat(frames)
Employees_list


# In[68]:


#Saving the employee list output
#Employees_list.to_csv('Output.csv', index=True)


# In[70]:


def get_input(input):
    koo=input
    koo=koo.drop(['Domain'], axis=1)
    koo=koo.drop(['Event1'], axis=1)
    koo=koo.drop(['Event2'], axis=1)
    koo = koo.reset_index()
    
    # Create an empty list 
    row_list =[] 
    # Iterate over each row 
    for index, rows in koo.iterrows(): 
        # Create list for the current row 
        my_list =[rows.Name] 
        
        row_list.append(my_list) 
        
    
    converted_list = [str(element) for element in row_list]
    joined_string = ",".join(converted_list)
    
    
    #text1= """
     #   Get a System Administration certification from PurpleHat today."""

    #text2= """
     #   Lockdown special courses on Ydemi. 22 hours left!
      #  """

    #text3="""
     #   CodeBoost codeathon is live now! Join the hackathon now!
      #  """

    #text4="""
     #   In the AI for Healthcare Nanodegree training program you'll leverage the power of AI to enable providers to deploy more precise, efficient, and impactful interventions at exactly the right moment in a patient’s care.

      #  In light of the worldwide COVID-19 pandemic, there has never been a better time to understand the possibilities of Artificial Intelligence within the healthcare industry and learn and to be trained how you can make an impact to better the world’s healthcare infrastructure.

       # The amount of data in healthcare has grown 20x in the past 7 years, causing an expected surge in the Healthcare AI market from $2.1 to $36.1 billion by 2025 at an annual growth rate of 50.4%. This increase in data will power the development and deployment of AI applications that enable the delivery of enhanced patient outcomes.
        #"""
    
    if input.equals(Input1_output)==True:
        text=text1
        
    elif input.equals(Input2_output)==True:
        text=text2
        
    elif input.equals(Input3_output)==True:
        text=text3
        
    else:
        text=text4
        
    
    
    hogo={'Column1':[text]}
    
    hogo=pd.DataFrame(hogo)
    
    hogo['Column2']=joined_string
    return hogo



    
    


# In[71]:


ans1=get_input(Input1_output)
ans2=get_input(Input2_output)
ans3=get_input(Input3_output)
ans4=get_input(Input4_output)
ans = [ans1,ans2,ans3,ans4]
Output_list=pd.concat(ans)
Output_list


# In[73]:


#Saving the output as another csv

Output_list.to_excel('output.xls', index=True)


# In[ ]:


#Random text check
#val=input()


# In[ ]:


#val


# In[ ]:


#textinput_domain=predict_from_text(val)
#textinput_event_type=predict_from_text2(val)
#print("Event output",':',"Domain-",textinput_domain,",","Event Type-", textinput_event_type)


# In[ ]:


#Employees that are interested
#Input_output=emp[(emp['Domain']==textinput_domain) & ((emp['Event1']==textinput_event_type) | (emp['Event2']==textinput_event_type))]
#print("Employee's Interested for Input1:")
#Input_output = Input_output.set_index('Name')
#Input_output


# In[ ]:


#Saving the employee list output
#Input_output.to_csv('Output1Exp.csv', index=True)


# In[ ]:


#Generating two column output
#ans1=get_input(Input_output)
#ans1


# In[ ]:


#Saving the output as another csv

#ans1.to_csv('Output2Exp.csv', index=True)


# In[ ]:




