import pandas as pd
data = pd.read_csv("yelp.csv")
data.isnull().sum()
data.useful.unique()
data.describe()
unnessary_columns = ["business_id","date","review_id","user_id"]
data.drop(unnessary_columns,inplace=True,axis=1)
data["length_of_text"] = data["text"].apply(len)

import matplotlib.pyplot as mlt
import seaborn as se
# se.scatterplot(data=data,hue="stars",x="funny",y="useful")
# mlt.show()
#
# se.pairplot(data=data,hue="stars")
# mlt.show()
#
# se.distplot(a=data.stars,bins=40)
# mlt.show()
# graph = se.FacetGrid(data=data,col='stars')
# graph.map(mlt.hist,'length_of_text',bins=50,color='blue')
# mlt.show()
# data.groupby("stars").mean()
# se.heatmap(data=data.corr(),annot=True)
# mlt.show()
def text_process(mess):
    import string as st
    from nltk.corpus import stopwords
    '''
    :param mess: string that needs to clean for further process
    :return:1. remove punc
		    2. remove stop word
		    3. return list of clean text words
    '''
    npunc = [p for p in mess if p not in st.punctuation]
    npunc = "".join(npunc)
    clean_mess = [word for word in npunc.split() if word.lower() not in stopwords.words("english")]
    return clean_mess
# print(data.text.apply(text_process))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def lets_try(train, labels):
    results = {}

    def test_model(clf):
        cv = KFold(n_splits=5, shuffle=True, random_state=45)
        r2 = make_scorer(r2_score)
        r2_val_score = cross_val_score(clf, train, labels, cv=cv, scoring=r2)
        scores = [r2_val_score.mean()]
        return scores

    clf = MultinomialNB()
    results["Multinomial"] = test_model(clf)

    clf = KNeighborsClassifier(1)
    results["KNN"] = test_model(clf)

    clf = SVC()
    results["SVC"] = test_model(clf)

    clf = DecisionTreeClassifier()
    results["Decision Tree"] = test_model(clf)

    clf = RandomForestClassifier()
    results["Random Forest"] = test_model(clf)

    results = pd.DataFrame.from_dict(results, orient='index')
    results.columns = ["R Square Score"]
    results.plot(kind="bar", title="Model Scores")
    axes = plt.gca()
    axes.set_ylim([0.5, 1])
    plt.show()
    return results

from sklearn.model_selection import train_test_split
t_train,t_test,l_train,l_test = train_test_split(data.text,data.stars,test_size = 0.25,random_state=10)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
pipl = Pipeline([
    ('bow',CountVectorizer(analyzer= text_process)),
    ('tfidf',TfidfTransformer()),
    ("multinomial",MultinomialNB())
])
pipl.fit(t_train,l_train)
pre = pipl.predict(t_test)
print(pre)