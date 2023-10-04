from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # data representation
import seaborn as sns
import nltk  # natural language tool kit
import praw  # reddit API wrapper for python
import credentials  # file containing keys and sensitive data
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import datetime
import matplotlib.pyplot as plt


class SearchResult:
    def __init__(self, title, date):
        self.title = title
        self.date = date


user_name = credentials.REDDIT_USERNAME
user_agent = f"Scraper 1.0 by {user_name}"
reddit = praw.Reddit(
    client_id=credentials.REDDIT_APIKEY,
    client_secret=credentials.REDDIT_SECRET,
    user_agent=user_agent
)

headlines = set()
for submission in reddit.subreddit('politics').search('Joe Biden'):
    obj = SearchResult(
        submission.title, datetime.datetime.fromtimestamp(submission.created_utc))
    headlines.add(obj)

sia = SIA()
results = []

for obj in headlines:
    polarity_score = sia.polarity_scores(obj.title)
    polarity_score['headline'] = obj.title
    polarity_score['date'] = obj.date
    results.append(polarity_score)

dataFrame = pd.DataFrame.from_records(results)

# making an average sentiment over each month
dateVsScore = dict()

for record in dataFrame.index:
    compoundScore = dataFrame['compound'][record]
    yearMonth = dataFrame['date'][record].to_period('M')
    if (yearMonth not in dateVsScore):
        dateVsScore[yearMonth] = [compoundScore, 1]
    else:
        dateVsScore[yearMonth] = [dateVsScore[yearMonth]
                                  [0] + compoundScore, dateVsScore[yearMonth][1] + 1]

for key in dateVsScore:
    dateVsScore[key] = [dateVsScore[key][0] / dateVsScore[key][1]]

df = pd.DataFrame.from_dict(
    dateVsScore, orient='index', columns=['Score'])

df.index = df.index.to_series().astype(str)

print(df.index)

plt.plot(df.index, df.Score)
plt.show()

# plt.plot(dataFrame.date, dataFrame.compound)
# plt.show()

# print(dataFrame.head())
