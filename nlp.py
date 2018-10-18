import nltk;
#nltk.download_shell();
messages=[line.rstrip() for line in open('SMSSpamCollection')];
print(len(messages));
import pandas as pd;
messages=pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message']);
print(messages.head());
import matplotlib.pyplot as plt;
import seaborn as sns;
messages['length']=messages['message'].apply(len);
messages['length'].hist(bins=50);
plt.show();
import string;
from nltk.corpus import stopwords;
def text_process(mess):
    no_punc=[char for char in mess if char not in string.punctuation];
    no_punc=''.join(no_punc);
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')];\

messages['message'].apply(text_process);
from sklearn.feature_extraction.text import CountVectorizer;
bow_transformer=CountVectorizer(analyzer=text_process).fit(messages['message']);
messages_bow=bow_transformer.transform(messages['message']);
print(messages_bow.shape);
print(messages_bow.nnz);
sparsity=(100.0*messages_bow.nnz)/(messages_bow.shape[0]*messages_bow.shape[1]);
print(sparsity);
from sklearn.feature_extraction.text import TfidfTransformer;
tfidf_transformer=TfidfTransformer().fit(messages_bow);
messages_tfidf=tfidf_transformer.transform(messages_bow);
from sklearn.naive_bayes import MultinomialNB;
spam_detect_model=MultinomialNB().fit(messages_tfidf,messages['label']);
predictions=spam_detect_model.predict(messages_tfidf);
from sklearn.cross_validation import train_test_split;
msg_train,msg_test,label_train,label_test=train_test_split(messages['message'],messages['label'],test_size=0.3,random_state=101);
from sklearn.pipeline import Pipeline;
pipeline=Pipeline([('bow',CountVectorizer(analyzer=text_process)),('tfidf',TfidfTransformer()),('classifier',MultinomialNB())]);
pipeline.fit(msg_train,label_train);
predict=pipeline.predict(msg_test);
from sklearn.metrics import classification_report;
print(classification_report(predict,label_test));





