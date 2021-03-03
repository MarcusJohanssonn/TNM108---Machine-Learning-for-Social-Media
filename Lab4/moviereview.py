import sklearn
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
# import CountVectorizer, nltk
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
# Now ready to build a classifier. 
# We will use Multinominal Naive Bayes as our model
from sklearn.naive_bayes import MultinomialNB

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix


moviedir = r'/Users/MarcusJohansson/Documents/Skola/TNM108/Lab4/movie_reviews'

# loading all files. 
movie = load_files(moviedir, shuffle=True)

print(len(movie.data))

# target names ("classes") are automatically generated from subfolder names
print(movie.target_names)

# First file seems to be about a Schwarzenegger movie. 
print(movie.data[0][:500])

# first file is in "neg" folder
movie.filenames[0]

# first file is a negative review and is mapped to 0 index 'neg' in target_names
movie.target[0]

# Split data into training and test sets

docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target, test_size = 0.20, random_state = 12)

# initialize CountVectorizer
movieVzer= CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000) # use top 3000 words only. 78.25% acc.
# movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)         # use all 25K words. Higher accuracy

# fit and tranform using training text 
docs_train_counts = movieVzer.fit_transform(docs_train)

# 'screen' is found in the corpus, mapped to index 2290
movieVzer.vocabulary_.get('screen')

# Likewise, Mr. Steven Seagal is present...
movieVzer.vocabulary_.get('seagal')

# huge dimensions! 1,600 documents, 3K unique terms. 
docs_train_counts.shape

# Convert raw frequency counts into TF-IDF values
movieTfmer = TfidfTransformer()
docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)


# Same dimensions, now with tf-idf values instead of raw frequency counts
docs_train_tfidf.shape


# Using the fitted vectorizer and transformer, tranform the test data
docs_test_counts = movieVzer.transform(docs_test)
docs_test_tfidf = movieTfmer.transform(docs_test_counts)


# Train a Multimoda Naive Bayes classifier. Again, we call it "fitting"
clf = MultinomialNB()
clf.fit(docs_train_tfidf, y_train)


# Predict the Test set results, find accuracy
y_pred = clf.predict(docs_test_tfidf)
sklearn.metrics.accuracy_score(y_test, y_pred)


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride', 
            'Steven Seagal was terrible', 'Steven Seagal shone through.', 
              'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through', 
              "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough', 
              'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

reviews_new_counts = movieVzer.transform(reviews_new)         # turn text into count vector
reviews_new_tfidf = movieTfmer.transform(reviews_new_counts)  # turn into tfidf vector


# have classifier make a prediction
pred = clf.predict(reviews_new_tfidf)

for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))


print(gs_clf.best_score_)
