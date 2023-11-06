#!pip install google-play-scraper
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google_play_scraper import app, Sort, reviews_all

# Read in existing dataset
df_existing = pd.read_csv('/content/drive/MyDrive/PFEFolder/sololearn.csv', sep=',', encoding='utf-8')

# Set last month as the cutoff date
cutoff_date = datetime.now() - timedelta(days=30)

# Scrape reviews for the last month and stop when data already exists
new_reviews = []
for review in reviews_all(
        'com.sololearn',
        sleep_milliseconds=0, # defaults to 0
        lang='en', # defaults to 'en'
        country='uk', # defaults to 'us'
        sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
    ):
    if review['at'] >= cutoff_date:
        if review['content'] in df_existing['content'].values:
            break
        new_reviews.append(review)
    else:
        break

# Convert new reviews to dataframe
df_new = pd.DataFrame(new_reviews, columns=['userName','content','score','at'])
df_new = df_new.sort_values(by='score', ascending=False)

# Append new reviews to existing dataset and save
df_all = pd.concat([df_existing, df_new], ignore_index=True)
df_all.to_csv("sololearn.csv", encoding='utf-8', index=False)