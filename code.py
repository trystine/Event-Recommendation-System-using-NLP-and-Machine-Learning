import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import punkt
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

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
    
#TF-IDF object
path_tfidf = "Pickles/tfidf.pickle"
with open(path_tfidf, 'rb') as data:
    tfidf = pickle.load(data)
    
#Another Classifier
path_tfidf2 = "Pickles/tfidf2.pickle"
with open(path_tfidf2, 'rb') as data:
    tfidf2 = pickle.load(data)

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

#Now let's write a function that tells us the category given the category code:
def get_category_name(category_id):
    for category, id_ in class_codes.items():    
        if id_ == category_id:
            return category

def get_category_name2(category_id):
    for category, id_ in event_codes.items():    
        if id_ == category_id:
            return category

#Finally, let's write a function that includes the whole process:
def predict_from_text(text):
    
    # Predict using the input model
    prediction_rfc = rfc_model.predict(create_features_from_text(text))[0]
    prediction_rfc_proba = rfc_model.predict_proba(create_features_from_text(text))[0]
    
    # Return result
    category_rfc = get_category_name(prediction_rfc)
    return category_rfc
 
#For second process
def predict_from_text2(text):
    
    # Predict using the input model
    prediction_rfc = rfc_model2.predict(create_features_from_text(text))[0]
    prediction_rfc_proba = rfc_model2.predict_proba(create_features_from_text(text))[0]
    
    # Return result
    category_rfc = get_category_name2(prediction_rfc)
    return category_rfc

#Reading Input file
input_text=pd.read_csv('input.csv')
input_text

text1= input_text.iloc[0]
text2=input_text.iloc[1]
text3=input_text.iloc[2]
text4=input_text.iloc[3]

print(text1)
print(text2)
print(text3)
print(text4)

text1_domain=predict_from_text(text1)
text1_event_type=predict_from_text2(text1)
text2_domain=predict_from_text(text2)
text2_event_type=predict_from_text2(text2)
text3_domain=predict_from_text(text3)
text3_event_type=predict_from_text2(text3)
text4_domain=predict_from_text(text4)
text4_event_type=predict_from_text2(text4)

print("Event1 output",':',"Domain-",text1_domain,",","Event Type-", text1_event_type)
print("Event2 output",':',"Domain-",text2_domain,",","Event Type-", text2_event_type)
print("Event3 output",':',"Domain-",text3_domain,",","Event Type-", text3_event_type)
print("Event4 output",':',"Domain-",text4_domain,",","Event Type-", text4_event_type)

#Now let us see the employees that are interested
#We are going to display the names of employees with matching domain and event type based on various inputs

Input1_output=pd.DataFrame(emp[(emp['Domain']==text1_domain) & ((emp['Event1']==text1_event_type) | (emp['Event2']==text1_event_type))])
Input2_output=emp[(emp['Domain']==text2_domain) & ((emp['Event1']==text2_event_type) | (emp['Event2']==text2_event_type))]
Input3_output=emp[(emp['Domain']==text3_domain) & ((emp['Event1']==text3_event_type) | (emp['Event2']==text3_event_type))]
Input4_output=emp[(emp['Domain']==text4_domain) & ((emp['Event1']==text4_event_type) | (emp['Event2']==text4_event_type))]
print("Employee's Interested for Input1:")
Input1_output = Input1_output.set_index('Name')
Input1_output

print("Employee's Interested for Input2:")
Input2_output = Input2_output.set_index('Name')
Input2_output

#No one so we will write No one as a name input
Input2_output = Input2_output.append({'Name' : 'No employee for text2'},ignore_index=True)
Input2_output = Input2_output.set_index('Name')
Input2_output

print("Employee's Interested for Input3:")
Input3_output = Input3_output.set_index('Name')
Input3_output

#No one so we will write No one as a name input
Input3_output = Input3_output.append({'Name' : 'No employee for text3'},ignore_index=True)
Input3_output = Input3_output.set_index('Name')
Input3_output

print("Employee's Interested for Input4:")
Input4_output = Input4_output.set_index('Name')
Input4_output

frames = [Input1_output,Input2_output,Input3_output,Input4_output]
Employees_list=pd.concat(frames)
Employees_list

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

ans1=get_input(Input1_output)
ans2=get_input(Input2_output)
ans3=get_input(Input3_output)
ans4=get_input(Input4_output)
ans = [ans1,ans2,ans3,ans4]
Output_list=pd.concat(ans)
Output_list

#Saving the output as another csv

Output_list.to_excel('output.xls', index=True)
