import nltk
from nltk.corpus.reader import wordnet
# nltk.download('stopwords')
# nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import pandas as pd
import json
from nltk.corpus import stopwords
from spacy import displacy
from nltk.tokenize import TweetTokenizer
from nltk.stem import  WordNetLemmatizer
import re


stop_words = set(stopwords.words("english"))
class Preprocess_Pipeline:

    # class that loads the json file dataset
    # it has its own lemmatizer and stemmer objects 
    # it uses a Tweet Tokenizer to tokenize the json dataset 
    # since the Tweet Tokenizer removes words like @
    def __init__(self,filename):
        #f=open(filename)
        #self.data=json.load(f)
        self.data=pd.read_pickle(filename)
        self.lemmatizer=WordNetLemmatizer()
        
        self.tokenized_word=[]
        self.tknzr=TweetTokenizer(strip_handles=True, reduce_len=True)
        self.final_lst=[]

    # tokenizes all the sentences in the answer/questions of the dataset and 
    # # remove all the punctuations from the tokenized list of words
    # remove the links from the questions
    # remove all the stop words from the tokenized questions
    # lemmatize each word in the question
    def tokenize_clean_data(self):
        self.data['question']=self.data.apply(lambda row:re.sub(r'^https?:\/\/.*[\r\n]*', '', row['question'], flags=re.MULTILINE), axis=1)
        self.data['question']=self.data.apply(lambda row:self.tknzr.tokenize(row['question']), axis=1)
        self.data['question']=self.data.apply(lambda row:[word for word in row['question'] if word.isalnum()], axis=1)
        self.data['question']=self.data.apply(lambda row:[word for word in row['question']if word not in stop_words],axis=1)
        self.data['question']=self.data.apply(lambda row: self.lemmatize_phrase(row['question']),axis=1)
        self.data['title']=self.data.apply(lambda row:self.tknzr.tokenize(row['title']), axis=1)
       
        self.data['title']=self.data.apply(lambda row:[word for word in row['title'] if word.isalnum()], axis=1)
        self.data['title']=self.data.apply(lambda row:[word for word in row['title']if word not in stop_words],axis=1)
        self.data['title']=self.data.apply(lambda row: self.lemmatize_phrase(row['title']),axis=1)

      

  
    # uses the class lemmatizer to lematize a list of strings
    def lemmatize_phrase(self,phrase):
        lst=nltk.pos_tag([word.lower() for word in phrase])
        
        lst= [self.lemmatizer.lemmatize(word[0],self.get_pos_tag(word[1])) for word in lst if self.get_pos_tag(word[1]) is not None]
       
        return lst

    # function that returns the wordnet pos tag of a specific
    # pos tag
     # gets the part of speech tagging of a word (whether verb, noun, adjective...)
    # this is a helper function that enables lemmatization by providing the type of word
   
    def get_pos_tag(self,tag):
        
        if tag in ['JJ', 'JJR', 'JJS']:
            return wordnet.ADJ
        elif tag in ['RB', 'RBR', 'RBS']:
            return wordnet.ADV
        elif tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            return wordnet.NOUN
        elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            return wordnet.VERB
        return None 

    def trigger_pipeline(self):
        self.tokenize_clean_data()
        result=self.data.loc[:,['title','main_cat','question','answer']]
        return result

    

process_obj= Preprocess_Pipeline('Code/dataset_final.pkl')
df_res=process_obj.trigger_pipeline()
print(df_res.head())
# result=pd.DataFrame(df_res,columns=['title','main_cat','question','answer'])
# print(result.head())