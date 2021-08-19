import pandas as pd
import nltk
import random
import string
import re

df = pd.read_csv('lyrics.csv')

df.drop(columns=['index', 'song', 'artist'], axis=1, inplace=True )

# Get rid of NAs in any column
df = df.dropna()

# Exclude songs with no genres
exclude = ['Not Available', 'Other', 'Jazz', 'Electronic']
df = df[~df.genre.isin(exclude)]
df['genre'].unique()
# maybe merge some of these genres together to improve generalization

# Collapse 'Rock' and 'Metal' into one category 'Rock'
df.loc[df.genre == 'Metal', 'genre'] = 'Rock'

# Collapse 'Country' and 'Folk' into one category 'Country'
df.loc[df.genre == 'Folk', 'genre'] = 'Country'

sample = df.sample(frac=0.5, replace=False, random_state=1)

lyrics = pd.Series.tolist(sample['lyrics'])

# want to convert to lowercase and remove all special characters
for i,lyric in enumerate(lyrics):
    lyric = lyric.lower()
    lyrics[i]= re.sub(r"[^a-zA-Z0-9]+", ' ', lyric)
#print(lyrics)

from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
#print(stop_words)

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist

cleanlyrics = []
wordcount = []
uniquewordcount = []

for i,l in enumerate(lyrics):
    # tokenize words in each song lyric
    tokenized_word= word_tokenize(l)
    
    # remove stop words
    filtered = []
    for w in tokenized_word:
        if w not in stop_words:
            filtered.append(w)
    #   print filtered
    
    # Lemmatize words
    lemmatized = []
    lem = WordNetLemmatizer()
    for w in filtered:
        lemmatized.append(lem.lemmatize(w,"v"))
    #print lemmatized

    #    delete all occurences of intro, chorus, verse, bridge...
    exclude = ['intro', 'chorus', 'verse', 'bridge']
    lemmatized = [value for value in lemmatized if value not in exclude]

    # create new list to series for wordcount
    uniquewordcount.append(len(list(set(lemmatized))))

    # create new list to series for wordcount
    wordcount.append(len(lemmatized))
    
    #cleanlyrics.append(top)
    cleanlyrics.append(" ".join(lemmatized))

sample['lyrics'] = cleanlyrics
sample['wordcount'] = wordcount
sample['uniquewordcount'] = uniquewordcount
#print len(sample)
sample = sample.dropna()
#print len(sample)

# do not include songs with word count less than a hundred
sample = sample[sample['wordcount'] >= 50]

# write to csv
sample.to_csv('cleaned_lyrics.csv')


# Partition dataframe into training and testing

# Get no. of training and testing
p = 0.8
train = int(p*len(sample['lyrics']))
print train
test = len(sample['lyrics']) - train
print test

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(sample['lyrics'])

from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer()
text_tf= tf.fit_transform(sample['lyrics'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                                                    text_tf, sample['genre'], test_size=0.3, random_state=123)

#%% MULTINOMIAL NB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))

#%% GAUSSIAN NB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train.todense(), y_train)

# making predictions on the testing set
y_pred = gnb.predict(X_test.todense())

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

#%% BERNOULLI NB
bnb = BernoulliNB()
bnb.fit(X_train,y_train)
accuracy = bnb.score(X_test,y_test)
print "accuracy for bernoulli naive bayes: %s"%(accuracy)

#%% LOGISTIC REGRESSION
lr = LogisticRegression(solver="liblinear", multi_class="ovr")
lr.fit(X_train,y_train)
print "accuracy for LogisticRegression: %s"%(lr.score(X_test,y_test))


#%% SUPPORT VECTOR MACHINES
from sklearn import svm

clf = svm.SVC(gamma = 0.001, C= 100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
