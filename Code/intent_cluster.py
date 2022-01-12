from nltk import tag
import pandas as pd
import nltk
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from doc2vec import Doc2VecModel,create_corpus,intents

# retrieve the saved model from the pickle file
# retrieve the tagged document corpus of the model
# get the list of intents from the doc2vec
# for each intent key, get the top 1000 most similar vector documents 
# retrieve those docs from tagged docs corpus by using the index of the vecdoc 
# create intent file dataset and save to yaml

print("here in cluster")
model=Doc2Vec.load("Models/Doc2Vec.model")
corpus=create_corpus('Datasets/required_dataset.pkl')
tagged_doc=[TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(corpus)]

# function that retrieves documents from the corpus based on the tags and groups them
# according to their respective intents
# returns a pandas dataframe as training data
def generate_training_data():
    training_dict={}
    for key in intents:
        doc_lst=[]
        inferred_vector=model.infer_vector(nltk.word_tokenize(intents[key]))
        sims=model.dv.most_similar(inferred_vector,topn=1000)
        for item in sims:
            index=int(item[0])
            doc_lst.append(tagged_doc[index][0])
        training_dict[key]=doc_lst
        df_train=pd.DataFrame(list(dict.items()),columns=['Intents', 'Documents'])
    return df_train




