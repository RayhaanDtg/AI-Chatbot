
import pandas as pd
import nltk
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# intents file as a dictionary with intent tag and key words associated with each intent
intents={
     
     
    "Refund": "policy refund return change money back",
    "Payment" : "payment credit card bank mastercard visa",
    "Shipping": "delivery ship product pack how long package",
    "storage": "storage disk hard drive memory save file format how",
    "Order": "purchase buy order past available",
    "Display": "cable work macbook nook monitor screen card slot driver adapter graphic lenses fit mount camera flash frame cell phone tablet",
    "Charging": "cable plug charge compatible port charger",
    "Connection": "wifi compatible jack plug able wireless phone android iphone antenna bluetooth",
    "Product_description": "price information colour size inch width height fit cover do have specification sales discount how much money water proof warranty model",
    "Issue" : "issue not work properly cannot fix update",
    "Greeting": "hello hi how be you what up morning"
}

# function that add the intent values  to the required questions of the dataset based on the categories
# get the all the rows of the dataset where main_cat = ['Computers', 'Camera & Photo','Camera &amp;' ,'Cell Phones &amp; Accessories']
def create_corpus(filename):
    df=pd.read_pickle(filename)
    condition=(df['main_cat']=="Camera & Photo") | (df['main_cat']=="Camera &amp; Photo") | (df['main_cat']=="Cell Phones &amp; Accessories") | (df['main_cat']=="Cell Phones & Accessories") | (df['main_cat']=="Computers")
    
    corpus=df.loc[condition].loc[:,'question']
    intent_df=pd.DataFrame(data=intents.values(),columns=['question'])
    intent_df_tkzr=intent_df.apply(lambda row: nltk.word_tokenize(row['question']),axis=1)
    
   
    final_corpus= pd.concat([corpus,pd.Series(intent_df_tkzr)],axis=0)
    return final_corpus



 
   

# class of a Doc2Vec deep learning model
# takes in corpus, number of epochs, vector size of embedding and learning rate as parameter
# the minimum learning rate for the model is 0.00025, which is what it will drop to as learning progresses
# the param dm=1 states that the algorithm to train is PV-DM 
# the min_count is 2 which states that ignore any words that appear only twice in doc
class Doc2VecModel:

    # creates the tagged doc from the corpus
    # initialise model
    # build vocabulary of model from corpus of tagged document
    def __init__(self,epoch,corpus,learning_rate,vec_size):
        self.tagged_doc=[TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(corpus)]
        self.model=Doc2Vec(vector_size=vec_size,
                        min_count=2,
                        alpha=learning_rate,
                        min_alpha=0.00025,
                        epochs=epoch,
                        dm=1)
        self.model.build_vocab(self.tagged_doc)
        self.epoch=epoch

    
    # trains the model 
    def train_model(self,pathname):
        
        
        print("starting model")    
        self.model.train(self.tagged_doc,total_examples=self.model.corpus_count,epochs=self.model.epochs)
        print('training done')
           
        self.model.save(pathname)
        print("model saved")
        return None


# final_corpus=create_corpus('Datasets/required_dataset.pkl')
# model=Doc2VecModel(epoch=100,corpus=final_corpus,learning_rate=0.025,vec_size=30)
# model.train_model('Models/Doc2Vec_2.model')