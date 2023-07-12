from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # data representation
import seaborn as sns
import nltk  # natural language tool kit
import praw  # reddit API wrapper for python
import credentials  # file containing keys and sensitive data
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

user_name = credentials.REDDIT_USERNAME
user_agent = f"Scraper 1.0 by {user_name}"
reddit = praw.Reddit(
    client_id=credentials.REDDIT_APIKEY,
    client_secret=credentials.REDDIT_SECRET,
    user_agent=user_agent
)

headlines = set()
for submission in reddit.subreddit('politics').hot(limit=25):
    headlines.add(submission.title)

dataFrame = pd.DataFrame(headlines)

dataFrame.to_csv('headlines.csv', header=False, encoding='utf-8', index=False)

sia = SIA()
results = []

for line in headlines:
    polarity_score = sia.polarity_scores(line)
    polarity_score['headline'] = line
    results.append(polarity_score)

dataFrame = pd.DataFrame.from_records(results)

print(dataFrame.head())
