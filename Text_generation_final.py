
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data
import pandas as pd

# Load a text file if required
# All of the sentences should end with a full stop. This is a criteria for defining the end of a sentence.
# If the sentence is not ending with a full stop, this will not work as expected
raw_data = pd.read_csv("syn_rep_data.csv")
data = raw_data['sentence']
output_list = []
for x in range(0,len(data)):
    text = data[x]
#text = "the samsung galaxy s8 plus is an awesome phone."
#output = ""

# Load the pretrained neural net
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #for i in range(0,len(data)):
    
    # Tokenize the text
    tokenized = tokenizer.tokenize(text)

    # Get the list of words from the entire text
    words = word_tokenize(text)

    # Identify the parts of speech
    tagged = nltk.pos_tag(words)
    
    output = []
    # This loop is for the number of sentences that has to be generated 
    # The range value can be increased or decreased accordingn to your need
    for a in range(0,5):


        for i in range(0,len(words)):
            replacements = []

        # Only replace nouns with nouns, vowels with vowels etc.
            for syn in wordnet.synsets(words[i]):

                # Do not attempt to replace proper nouns or determiners or Pronoun Possessive or 
                if tagged[i][1] == 'NNP' or tagged[i][1] == 'DT' or tagged[i][1] == 'PRP' or tagged[i][1] == 'VBZ':
                    break

                # The tokenizer returns strings like NNP, VBP etc
                # but the wordnet synonyms has tags like .n.
                # So we extract the first character from NNP ie n
                # then we check if the dictionary word has a .n. or not 
                word_type = tagged[i][1][0].lower()
                #print(word_type)
                if syn.name().find("."+word_type+"."):
                    # extract the word only
                    r = syn.name()[0:syn.name().find(".")]
                    replacements.append(r)
            #print(replacements)
            #for j in range(0,5):
            if len(replacements) > 0:
                # Choose a random replacement
                replacement = replacements[randint(0,len(replacements)-1)]
                #print(replacement)

                output.append(replacement)

            #output[]
            else:
                # If no replacement could be found, then just use the
                # original word
                output.append(words[i])
    output_list.append(output)
        
        


# In[2]:


len(output_list)


# In[3]:


a = []
b = []
a_a = []
count = 0
dot = "."
for x in range(0,len(output_list)):
    
    for i in output_list[x]:
        if i != dot:
            a.append(i)
        else:
            b.append(a)
            a = []
    a_a.append(b)
    b = []


# In[4]:


nltk.download('perluniprops')
from nltk.tokenize.moses import MosesDetokenizer

detokenizer = MosesDetokenizer()
de_tok = []
detok_desc = []
final_list = []
for x in range(0,len(a_a)):
    for i in a_a[x]:
        de_tok = detokenizer.detokenize(i, return_str=True)
        detok_desc.append(de_tok)
    final_list.append(detok_desc)
    detok_desc = []


# In[5]:


final_list[0]


# In[6]:


data[0]


# In[7]:


final_list

#print(*final_list, sep = "\n")
#print('\n'.join(map(str, final_list)))


# In[8]:


import gensim, logging
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import time
import gensim


# In[9]:


import gensim.models.keyedvectors as word2vec
sg_ = 1 # the training algorithm. If sg=0, CBOW is used. Otherwise (sg=1), skip-gram is employed.
alg = 'CBOW' if sg_ == 0 else 'sg'

model=word2vec.KeyedVectors.load_word2vec_format('/Users/sourodip/Downloads/GoogleNews-vectors-negative300.bin.gz',binary=True)
words = list(model.wv.vocab.keys())
print(f"The number of words: {len(words)}")


# In[10]:


txt_list=raw_data['sentence'].values.tolist()
print(txt_list)


# In[11]:


txt_gen=pd.read_csv('Generated_sentence.txt',encoding='latin-1')
txt_list_gen=txt_gen['sentence'].values.tolist()
txt_list_gen


# In[32]:


filter_gen=[]
for i in txt_list:
    for j in txt_list_gen:
        distance = model.wmdistance(i,j)
        if distance <0.5:
            filter_gen.append(j)
            print("Original Sentence:",i)
            print("Generated Sentence:",j)
            print("WMD:",distance)
        else:
            continue
      


# In[53]:


from textblob import TextBlob
filter_gen2=[]
for x in filter_gen:
    gen_sen = TextBlob(x)
    if gen_sen.sentiment.polarity<0.3 or gen_sen.sentiment.polarity<-0.3:
        print('The sentence:',x)
        print('polarity:',gen_sen.sentiment.polarity)
        print('subjectivity:',gen_sen.sentiment.subjectivity)
        filter_gen2.append(x)
    else:
        continue


# In[40]:


import language_check
filter_gen3=[]
tool = language_check.LanguageTool('en-US')
for y in filter_gen2:
    if len(tool.check(y))<1:
        filter_gen3.append(y)
    else:
        continue


# In[54]:


filter_gen3


# In[38]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

filter_vader=[]

sid_obj = SentimentIntensityAnalyzer() 
for c in filter_gen:
    sentiment_dict = sid_obj.polarity_scores(c)
    if sentiment_dict['compound']<0.3 or sentiment_dict['compound']<-0.3:
        filter_vader.append(c)
    else:
        continue


# In[39]:


len(filter_vader)


# In[23]:


filter_vader


# In[50]:


import language_check
filter_gen_new=[]
tool = language_check.LanguageTool('en-US')
for y in filter_vader:
    if len(tool.check(y))<1:
        print(y)
        filter_gen_new.append(y)
    else:
        continue


# In[51]:


len(filter_gen_new)

