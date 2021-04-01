import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

df = pd.read_csv('yelp.csv')
# print(df.iloc[5])
# df.to_csv("yelp1.csv")

# print(df.isnull().sum())

# print(df[df['type'] !='review'])

# print(df.shape)
# print(df.describe())

df['text_length'] = df['text'].apply(len)
# print(df)

def text_process(mess):

	nonpunc = [c for c in mess if c not in string.punctuation]
	nonpunc  = "".join(nonpunc)
	clean_mess  =  [word for word in nonpunc.split() if word.lower() not in stopwords.words("english")]
	return clean_mess

# vec1 = CountVectorizer(analyzer= text_process).fit(df['text'])
# X = vec1.transform(df['text'])
# print(X.nnz)


x_train,x_test,y_train,y_test =  train_test_split(df['text'],df['stars'],test_size=.25,random_state=10)

pipeline = Pipeline([
	('bow',CountVectorizer(analyzer=text_process)),
	('tfidf',TfidfTransformer()),
	('classifier',MultinomialNB())
])
pipeline.fit(x_train, y_train)
predict = pipeline.predict(x_test)

print(classification_report(predict,y_test))