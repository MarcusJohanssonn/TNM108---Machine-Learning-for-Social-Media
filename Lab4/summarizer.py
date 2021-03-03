from summa.summarizer import summarize
from summa import keywords


text = ''' Hej Hej

'''


# Define length of the summary as a proportion of the text

print(summarize(text, words=100))
#summarize(text, words=50)

#print("Keywords:\n", keywords.keywords(text))
print("Top 3 Keywords:\n", keywords.keywords(text, words=3))







