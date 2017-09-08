
# coding: utf-8

# In[5]:

import os
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
get_ipython().magic(u'matplotlib inline')


# In[6]:

os.getcwd()


# In[7]:

os.chdir('A:\UNCC\Spring 2017\ML\Project\QuoraDataset')


# In[8]:

quora_dataset = pd.read_table('quora_duplicate_questions.tsv')
temp = quora_dataset


# In[9]:

quora_dataset


# In[10]:

quora_test_set = pd.read_csv('test.csv')


# In[186]:

quora_test_set


# In[11]:

Non_duplicate = temp[quora_dataset['is_duplicate'] == 1].count()


# In[12]:

Non_duplicate


# In[13]:

len(quora_dataset)


# In[14]:

#36.919
print('Duplicate pairs: {}%'.format(round(quora_dataset['is_duplicate'].mean()*100, 2)))


# In[15]:

#around 37% of question pairs are duplicate
#Brilliantly done...Mind you the last column is either 0 or 1...So you can easily get the fraction of the
#question pairs which are duplicate


# In[16]:

quora_dataset.groupby(['is_duplicate']).mean()


# In[17]:

import matplotlib.pyplot as plt


# In[18]:

import matplotlib
matplotlib.style.use('ggplot')


# In[ ]:




# In[19]:

quora_dataset.question1.str


# In[17]:

quora_dataset.isnull().sum()


# In[20]:

quora_dataset.isnull()


# In[ ]:




# In[21]:

null_values = quora_dataset.isnull().unstack()


# In[22]:

null_values


# In[25]:

t = null_values[null_values]


# In[26]:

t


# In[27]:

quora_test_set.isnull().sum()


# In[28]:

null_test_values = quora_test_set.isnull().unstack()


# In[29]:

quora_test_set[379204:379216]


# In[30]:

t1 =null_test_values[null_test_values]


# In[31]:

t1


# In[32]:

# Remove nan's


# In[33]:

quora_dataset = quora_dataset.dropna()


# In[34]:

quora_test_set = quora_test_set.dropna()


# In[35]:

quora_dataset.isnull().sum()


# In[36]:

quora_test_set.isnull().sum()


# In[37]:

#quora_test_set.plot(x='question1',y='question2')
#Requires numeric data


# In[38]:

quora_dataset.question1.str


# In[ ]:




# In[39]:

print(quora_dataset.question1.str.len())
print("Average number of characters in question set1:" + str(quora_dataset.question1.str.len().mean()))
print("Maximum number of characters in question set1:" + str(quora_dataset.question1.str.len().max()))
print("Minimum number of characters in question set1:" + str(quora_dataset.question1.str.len().min()))


# In[ ]:




# In[40]:

#For question 2 
print(quora_dataset.question2.str.len())
print("Average number of characters in question set2:" + str(quora_dataset.question2.str.len().mean()))
print("Maximum number of characters in question set2:" + str(quora_dataset.question2.str.len().max()))
print("Minimum number of characters in question set2:" + str(quora_dataset.question2.str.len().min()))


# In[41]:

#print("Average number of characters is {0}".format((quora_dataset.question2.str.len().mean())))


# In[42]:

EqualList = (quora_dataset.question1 == quora_dataset.question2).count()


# In[43]:

type(quora_dataset.question1)


# In[44]:

rows = list(quora_dataset[quora_dataset['question1'] == quora_dataset.question2].index)


# In[45]:

print(rows)


# In[46]:

quora_dataset.iloc[18914].question1


# In[47]:


quora_dataset.iloc[18914].question2


# In[48]:

quora_dataset.iloc[6242]


# In[49]:

quora_dataset.iloc[23507].question2


# In[50]:

quora_dataset.iloc[23507].question1


# In[51]:

quora_dataset.iloc[399339].question2


# In[52]:

quora_dataset.iloc[399340].question1


# In[53]:

quora_dataset['lengthQ1'] = quora_dataset.question1.apply(lambda x:len(str(x)))


# In[54]:

quora_dataset['lengthQ1']


# In[55]:

quora_dataset['lengthQ2'] = quora_dataset.question2.apply(lambda x:len(str(x)))


# In[56]:

quora_dataset['lengthQ2']


# In[57]:

quora_dataset['diff_len'] = quora_dataset.lengthQ1 - quora_dataset.lengthQ2


# In[58]:

quora_dataset['diff_len']


# In[59]:

quora_dataset.iloc[404265]


# In[60]:

quora_dataset['diff_len'] == 0


# In[61]:

#TO find those values where the difference is 0
quora_dataset.loc[quora_dataset['diff_len'] == 0]


# In[62]:

# By char length,it means the length of the unique characters present,so a set is taken


# In[63]:

quora_dataset['CharLengthQ1'] = quora_dataset.question1.apply(lambda x : len(''.join(set(str(x).replace(' ','')))))


# In[64]:

quora_dataset['CharLengthQ1']


# In[65]:

quora_dataset['CharLengthQ2'] = quora_dataset.question2.apply(lambda x : len(''.join(set(str(x).replace(' ','')))))


# In[66]:

diff_dataset = quora_dataset[['CharLengthQ2','CharLengthQ1']].copy()


# In[67]:

diff_dataset


# In[68]:

quora_dataset[['CharLengthQ2','CharLengthQ1']]


# In[69]:

quora_dataset.iloc[0]['question1']


# In[70]:

len("What is the story of Kohinoor (Koh-i-Noor) Diamond?")


# In[67]:

#It constructs only a set of characters and not set of words


# In[68]:

#Now to the word length


# In[71]:

quora_dataset['wordLengthQ2'] =  quora_dataset.question2.apply(lambda x : len(str(x).split()))


# In[72]:

quora_dataset['wordLengthQ1'] =  quora_dataset.question1.apply(lambda x : len(str(x).split()))


# In[73]:

#
diff_dataset = quora_dataset[['wordLengthQ2','wordLengthQ1','CharLengthQ2','CharLengthQ1']].copy()


# In[74]:

diff_dataset


# In[75]:

#quora_dataset['CommonWords'] =  quora_dataset.question2.apply(lambda x : set((str(x).split()))).intersection(quora_dataset.question1.apply(lambda x : set((str(x).split()))))


# In[76]:

quora_dataset['set2'] = quora_dataset.question2.apply(lambda x : set((str(x).lower().replace("?","").split())))
quora_dataset['set1'] = quora_dataset.question1.apply(lambda x : set((str(x).lower().replace("?","").split())))


# In[77]:

quora_dataset['set2']


# In[ ]:




# In[78]:

quora_dataset['set1']


# In[79]:

quora_dataset[:5]


# In[80]:

#quora_dataset.to_csv("R:\processed_quora.csv", sep='\t')


# In[81]:

quora_dataset['commonwords'] = quora_dataset.apply(lambda x:  (set((str(x['question1']).lower().replace("?","").split())).intersection( set((str(x['question2']).lower().replace("?","").split())))),axis =1)


# In[82]:

quora_dataset[['commonwords','question1','question2']]


# In[ ]:




# In[83]:

#question ids


# In[84]:

qids = pd.Series(quora_dataset['qid1'].tolist() + quora_dataset['qid2'].tolist())


# In[85]:

len(qids)


# In[86]:

print("Number of unique questions:")


# In[87]:

print(len(np.unique(qids)))


# In[88]:

plt.figure(figsize=(12, 5))
plt.hist(qids.value_counts(), bins=50)
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')


# In[89]:

qids.value_counts()


# In[90]:

Count1 = quora_dataset['qid1'].value_counts() 


# In[91]:

type(Count1)


# In[92]:

Count2 = quora_dataset['qid2'].value_counts() 


# In[93]:

Count2


# In[94]:

from sklearn.preprocessing import StandardScaler


# In[ ]:




# In[ ]:




# In[95]:

quora_inputFeatures1 = quora_dataset[['lengthQ1','lengthQ2','diff_len','CharLengthQ1','CharLengthQ2','wordLengthQ2','wordLengthQ1','is_duplicate']]


# In[96]:

quora_inputFeatures1[0:6]


# In[97]:

features = quora_inputFeatures1.columns[:-1]


# In[98]:

features


# In[99]:

X = quora_inputFeatures1[features]


# In[100]:

X


# In[101]:

Y = quora_inputFeatures1['is_duplicate']


# In[102]:

Y


# In[103]:

from sklearn.preprocessing import StandardScaler


# In[104]:

scaler = StandardScaler()


# In[105]:

X_train = scaler.fit_transform(X)


# In[106]:

X_train


# In[107]:

Y


# In[108]:

#using logistic Regression
#Test and training split


# In[109]:

from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# In[110]:

XCross_train, XCross_test, y_Crosstrain, y_Crosstest = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[ ]:




# In[111]:

logistic = linear_model.LogisticRegression()


# In[112]:

logistic


# In[113]:

logistic.fit(XCross_train,y_Crosstrain)


# In[114]:

scores = cross_val_score(logistic,XCross_test,y_Crosstest, cv=5)


# In[115]:

scores


# In[116]:

y_predicted_class = logistic.predict(XCross_test)
print(metrics.accuracy_score(y_Crosstest,y_predicted_class))


# In[117]:

def logisticGridSearch():
    
    tuned_parameters = [{
                         'solver' :["newton-cg","lbfgs","liblinear","sag"],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                        }   
                       ]
   
    
    scores = ['precision', 'recall']
    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(linear_model.LogisticRegression(n_jobs = -1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
        clf.fit(XCross_train, y_Crosstrain)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_Crosstest, clf.predict(XCross_test)
        print(classification_report(y_true, y_pred))
        print()


# In[118]:

logisticGridSearch()


# In[ ]:

linear_model.LogisticRegression().get_params().keys()


# In[ ]:

linear_model.LogisticRegression(n_jobs = -1)


# In[ ]:




# In[119]:

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import string
import gensim, logging


# In[120]:

punlist = []
for pun in string.punctuation:
    punlist.append(pun)
print(punlist)
stops = set(stopwords.words("english") + punlist) 
#stops = set(stopwords.words("english") + string.punctuation) 


# In[121]:

stops


# In[122]:

model = gensim.models.KeyedVectors.load_word2vec_format('R:\DataSet\GoogleNews-vectors-negative300.bin', binary=True)  


# In[ ]:

skipcount = 0
VectorSent = []


# In[ ]:

def sent_to_vec(s):
    global skipcount
    sent = str(s).decode('utf-8').lower()
    #print("Words are: " + sent)
    words = word_tokenize(sent)
    #print("Tokenized words:")
    #print(words)
    words = [w for w in words if not w in stops]
    # print("After removing stopwords")
    #print(words)
    words = [w for w in words if w.isalpha()]
    #print("Taking only characters")
    #print(words)
    M = []
    for w in words:
        try:
                M.append(model[w])
                #print("Entering try")
                #print(model[w])            
        except  Exception, e:
                print "Couldn't do it: %s" % e
                skipcount = skipcount + 1
                continue
    VectorSent = M
    M = np.array(M)
    VectorSent = M
    print(len(VectorSent))
    for a in VectorSent:
        print a
        print(len(a))
    
    v = M.sum(axis = 0)
    print(v)
    return v/(np.sqrt(v**2)).sum()


# In[ ]:

a = "This monitor works beautifully. It is so clear that we can hear the baby breathing while he is sleeping. I definitely recommend this product to anyone looking for a good baby monitor."


# In[ ]:

a


# In[ ]:

vect = sent_to_vec(a)


# In[ ]:

vect


# In[ ]:

len(vect)


# In[ ]:

count = 0


# In[ ]:

a =
        


# In[ ]:

def questions_in_dataset():
    j=0
    SentenceVectors1 = []
    for i in quora_dataset.index:        
        if( (j+1)%1000 == 0 ):
            print "Question Processed %d Completed \n" % ( j+1)   
        SentenceVectors1.append(helper_sentence_to_vectors( quora_dataset["question1"][i] ) ) 
        j = j+1
    
    print("Question1-Processing Completed:")
    import pickle
    file_Name1 = "question1"
    fileObject1 = open(file_Name1,'wb') 
    pickle.dump(SentenceVectors1,fileObject1)
        
    j=0
    SentenceVectors2 = []
    for i in quora_dataset.index:        
        if( (j+1)%1000 == 0 ):
            print "Question Processed %d Completed \n" % ( j+1)   
        SentenceVectors1.append(helper_sentence_to_vectors( quora_dataset["question2"][i] ) ) 
        j = j+1
    
    
    print("Question2-Processing Completed:")
    import pickle
    file_Name2 = "question2"
    fileObject2 = open(file_Name2,'wb') 
    pickle.dump(SentenceVectors2,fileObject2)
        
    
   


# In[ ]:

questions_in_dataset()


# In[172]:

quora_dataset["question1"][7737]


# In[162]:

quora_dataset["question1"][2805]


# In[159]:

quora_dataset["question2"][2806]


# In[ ]:

content = content.decode('utf-8')


# In[190]:

sent_to_vec(quora_dataset["question1"][2806])


# In[ ]:

a = 10


# In[ ]:

a


# In[ ]:

stops


# In[208]:

# Use this to pickle the values
# import pickle
# newfile = 'mypickle.pk'
# with open(newfile, 'rb') as fi:
#   TestVar = pickle.load(fi) 
# clean_train_reviews = TestVar


# In[ ]:




# In[ ]:




# In[ ]:




# In[209]:

#splitting into testing training


# In[ ]:

XCross_train, XCross_test, y_Crosstrain, y_Crosstest = train_test_split(X, Y, test_size=0.75, random_state=42)


# In[ ]:

len(XCross_train)


# In[ ]:

len(XCross_test)


# In[ ]:

count = np.bincount(y_Crosstrain)


# In[ ]:

count


# In[ ]:

from sklearn.datasets import make_classification
from imblearn.under_sampling import     ClusterCentroids 
from collections import Counter


# In[ ]:


y = quora_dataset['is_duplicate']


# In[ ]:

y


# In[ ]:

QuoraInput = quora_dataset.ix[:, quora_dataset.columns != 'is_duplicate']


# In[ ]:

X = QuoraInput


# In[ ]:

Y = quora_dataset['is_duplicate']


# In[ ]:

print('Original dataset shape {}'.format(Counter(Y)))


# In[ ]:

cc = ClusterCentroids(random_state=42)


# In[ ]:

cc


# In[123]:

type(X)


# In[ ]:

type(Y)


# In[ ]:

msk = np.random.rand(len(quora_dataset)) < 0.4


# In[124]:

msk


# In[ ]:

QuoraSplitTrain = quora_dataset[msk]


# In[ ]:

QuoraSplitTest = quora_dataset[~msk]


# In[ ]:

len(QuoraSplitTrain)


# In[ ]:

len(QuoraSplitTest)


# In[ ]:

count = np.bincount(QuoraSplitTrain['is_duplicate'])


# In[ ]:

count


# In[ ]:

count = np.bincount(QuoraSplitTest['is_duplicate'])


# In[ ]:

count


# In[ ]:




# In[ ]:

SentenceVectors1 = []
SentenceVectors2 = []
def questions_in_dataset():
    j=0
   
    for i in QuoraSplitTrain.index:        
        if( (j+1)%1000 == 0 ):
            print "Question Processed %d Completed \n" % ( j+1)   
        SentenceVectors1.append(helper_sentence_to_vectors( QuoraSplitTrain["question1"][i] ) ) 
        SentenceVectors2.append(helper_sentence_to_vectors( QuoraSplitTrain["question2"][i] ) )
        j = j+1
    
    print("Question1,2-Processing Completed:")
    
        
    


# In[ ]:

def questions2_in_dataset():
    j=0
    SentenceVectors2 = []
    for i in QuoraSplitTrain.index:        
        if( (j+1)%1000 == 0 ):
            print "Question Processed %d Completed \n" % ( j+1)   
        SentenceVectors2.append(helper_sentence_to_vectors( QuoraSplitTrain["question2"][i] ) ) 
        j = j+1
    
    
    print("Question2-Processing Completed:")
    


# In[ ]:

def pickle_question1(SentenceVectors1):   
    import pickle
    file_Name1 = "question1"
    fileObject1 = open(file_Name1,'wb') 
    pickle.dump(SentenceVectors1,fileObject1)


# In[ ]:

def pickle_question2(SentenceVectors2):   
    import pickle
    file_Name2 = "question2"
    fileObject2 = open(file_Name2,'wb') 
    pickle.dump(SentenceVectors2,fileObject2)


# In[312]:

len(QuoraSplitTrain)


# In[131]:

newfile = 'question1'
with open(newfile, 'rb') as fi:
  TestVar = pickle.load(fi) 
question1_vectors = TestVar



# In[133]:

newfile = 'question2'
with open(newfile, 'rb') as fi:
  TestVar = pickle.load(fi) 
question2_vectors = TestVar


# In[129]:

question2_vectors


# In[132]:

question1_vectors


# In[342]:

len(question2_vectors)


# In[ ]:




# In[ ]:




# In[323]:

QuoraSplitTrain.to_pickle("Train")


# In[324]:

QuoraSplitTest.to_pickle("Test")


# In[343]:

QuoraSplitTrain.to_csv("SplitTrain.csv", sep='\t')


# In[344]:

QuoraSplitTest.to_csv("SplitTest.csv", sep='\t')


# In[328]:

len(quora_test_set)


# In[125]:

QuoraSplitTrain = pd.read_pickle('Train')


# In[126]:

QuoraSplitTest = pd.read_pickle('Test')


# In[127]:

QuoraSplitTrain


# In[128]:

QuoraSplitTest


# In[134]:

#######Fuzzy Features#####


# In[137]:

from fuzzywuzzy import fuzz


# In[135]:

##data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
##data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
##data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)##
##data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
##data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
##data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
##data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


# In[ ]:




# In[141]:

QuoraSplitTrain['QRatio'] = QuoraSplitTrain.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)


# In[142]:

QuoraSplitTrain['WRatio'] = QuoraSplitTrain.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)


# In[144]:

QuoraSplitTrain['partial_token_set_ratio'] = QuoraSplitTrain.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)


# In[ ]:




# In[146]:

QuoraSplitTrain['partial_token_sort_ratio'] = QuoraSplitTrain.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


# In[148]:

QuoraSplitTrain['partial_ratio'] = QuoraSplitTrain.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)


# In[ ]:

#fuzz_token_set_ratio


# In[ ]:

QuoraSplitTrain['fuzz_token_set_ratio'] = QuoraSplitTrain.apply(lambda x: fuzz.fuzz_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)


# In[ ]:

#fuzz_token_sort_ratio


# In[ ]:




# In[151]:

QuoraSplitTrain['fuzz_token_sort_ratio'] = QuoraSplitTrain.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


# In[152]:

QuoraSplitTrain['fuzz_token_sort_ratio']


# In[153]:

QuoraSplitTrain


# In[155]:

quora_inputFuzzyFeatures = QuoraSplitTrain[['fuzz_qratio','QRatio','WRatio','partial_token_set_ratio','partial_token_sort_ratio','partial_ratio','fuzz_token_sort_ratio','is_duplicate']]


# In[156]:

quora_inputFuzzyFeatures


# In[158]:

features = quora_inputFuzzyFeatures.columns[:-1]


# In[160]:

X = QuoraSplitTrain[features]


# In[161]:

Y = QuoraSplitTrain[]


# In[176]:

Y = QuoraSplitTrain['is_duplicate']


# In[177]:

Y


# In[178]:

XCross_train, XCross_test, y_Crosstrain, y_Crosstest = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[172]:

len(Y)


# In[179]:

X_train = scaler.fit_transform(XCross_train)


# In[180]:

X_train


# In[ ]:

X_test = scaler.transform(XCross_test)


# In[181]:

y_Crosstrain


# In[182]:

from sklearn import svm


# In[183]:

clf = svm.SVC()


# In[184]:

clf


# In[186]:

clf.fit(X_train,y_Crosstrain)


# In[187]:

Y_predict_class = clf.predict(XCross_test)


# In[189]:

from sklearn import metrics
from sklearn.metrics import confusion_matrix
metrics.accuracy_score(y_Crosstest,Y_predict_class)


# In[192]:

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


# In[193]:

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,1e-2,1e-1],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000],'gamma': [1e-3, 1e-4,1e-2,1e-1]},
                    {'kernel': ['poly'], 'C': [1, 10, 100, 1000],'degree':[2,5],'coef0':[0,2],'gamma': [1e-3, 1e-4,1e-2,1e-1]}
                   ]

scores = ['precision', 'recall']


# In[ ]:

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(XCross_train, y_Crosstrain)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_Crosstest, clf.predict(XCross_test)
    print(classification_report(y_true, y_pred))
    print()


# In[ ]:




# In[ ]:

quora_inputFeatures1 = quora_dataset[['lengthQ1','lengthQ2','diff_len','CharLengthQ1','CharLengthQ2','wordLengthQ2','wordLengthQ1','is_duplicate']]


# In[ ]:

quora_AllFeatures = QuoraSplitTrain[['QRatio','WRatio','partial_token_set_ratio','partial_token_sort_ratio','partial_ratio','fuzz_token_sort_ratio','lengthQ1','lengthQ2','diff_len','CharLengthQ1','CharLengthQ2','wordLengthQ2','wordLengthQ1','is_duplicate']]


# In[ ]:



