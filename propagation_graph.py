import os
import json
from nltk.sentiment.vader import VaderConstants
import pandas as pd
from utils import *
import networkx as nx
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class PropagationGraphBuilder():
    '''
    Each tweet is assumed to be in seperate json file in tweets folder and retweets for all tweets are in single json file named retweets.json
    '''

    def __init__(self, news_id, news_path, retweet_gap, vader: SentimentIntensityAnalyzer) -> None:
        self.news_path = news_path
        self.news_id = news_id
        self.retweet_gap = retweet_gap
        self.vader = vader
        self.num_of_tweets = None
        self.all_retweets = None
        self.tweets = None
        self.graph = None

    def build_edges(self, save_to_file, save_location) -> list:
        '''
        Builds edges of propagation graphs, given tweets and their retweets.
        '''

        # Build news node
        news_node = json.load(os.path.join(self.news_path, 'news_article.json'))

        # Sorting tweets based on timestamp
        tweets = self.collect_all_tweets()
        if not tweets:
            return
        tweets = sorted(tweets, key=lambda x: x['created_at'])

        # Searching immediate source of tweet and updating edges list
        edges = [] # List of tuples -> (ParentNode, ChildNode)
        edges.append((self.news_id, tweets[0]['id_str'])) # Source of earliest tweet will be news article itself

        ## Searching for rest of tweets
        for i in range(1, len(tweets)):
            target_tweet = tweets[i]
            source = None
            for j in range(0, i): 
                candidate_source = tweets[j]
                
                # Check if user of candidate_source mentions OR is mentioned by user target tweet
                cond_a = candidate_source['user']['id_str'] in [x['id_str'] for x in target_tweet['entities']['user_mentions']]
                cond_b = target_tweet['user']['id_str'] in [x['id_str'] for x in candidate_source['entities']['user_mentions']]

                if cond_a or cond_b:
                    source = candidate_source
                    break
            
            if source is None:
                edges.append((self.news_id, target_tweet['id_str']))
            else:
                edges.append((source['id_str'], target_tweet['id_str']))
            
            # Repeat the process for retweets
            retweets = self.collect_all_retweets(target_tweet)
            
            if retweets:
                retweets = sorted(retweets, key=lambda x: x['created_at'])
                edges.append((target_tweet['id_str'], retweets[0]['id_str']))
                for i in range(1, len(retweets)):
                    target_retweet = retweets[i]
                    source = None
                    for j in range(0, i):
                        candidate_source = retweets[j]

                        # 1. Check if user of candidate_source mentions user of target tweet
                        # 2. Check if target target was published within n hours of candidate source
                        cond_a = target_retweet['user']['id_str'] in [x['id_str'] for x in target_tweet['entities']['user_mentions']]
                        cond_b = ((target_retweet['created_at'] - candidate_source['created_at'])/3600) < self.retweet_gap 

                        if cond_a or cond_b:
                            source = candidate_source
                            break

                    if source is None:
                        edges.append((target_tweet['id_str'], target_retweet['id_str']))
                    else:
                        edges.append((source['id_str'], target_retweet['id_str']))

        # save edges to txt file
        if save_to_file:
            with open(os.path.join(save_location, f'{self.news_id}_graph.txt'), 'w') as f:
                json.dump(edges, f)

        # Create networkx graph with tweets/retweets as nodes
        # and keys of tweet/retweet object as node attributes for easy access 
        # when building node features
        g = nx.DiGraph()
        g.add_node(self.news_id, **news_node)
        g.add_nodes_from([
            (x['id_str'], x) for x in tweets
        ])
        g.add_edges_from(edges)
        self.graph = g

        return edges


    def build_node_features(self) -> pd.DataFrame:
        '''
        Returns a pandas dataframe of node-level features
        '''

        feature_names = [
            'source_time_diff'
            'account_age',
            'parent_time_diff'
            'avg_child_time_diff',
            'is_verified',
            'num_friends',
            'num_followers',
            'num_hastags',
            'num_mentions',
            'vader_score'
        ]
        data = []
        
        tweets = self.collect_all_tweets()
        source = self.graph.nodes[self.news_id] if self.graph.nodes[self.news_id]['publish_date'] else tweets[0]
        for tweet in tweets:

            id = tweet['id_str']
            
            # Temporal Features
            source_time_diff = tweet['created_at'] - source['publish_date']
            account_age = get_account_age(tweet['created_at'], tweet['user']['created_at'])
            parent_time_dff = get_parent_time_diff(self.graph, self.news_id, tweet['id_str'], tweet['created_at'])
            avg_child_time_diff = get_avg_child_time_diff(self.graph, tweet['id_str'], tweet['created_at'])

            # User-based features
            is_verified = 1 if tweet['user']['verified'] else 0
            num_friends = tweet['user']['friends_count']
            num_followers = tweet['user']['followers_count']
            
            # Text-based features
            num_hastags = len(tweet['entities']['hashtags'])
            num_mentions = len(tweet['entities']['user_mentions'])
            vader_score = get_vader_score(tweet['text'])

            data.append([
                id, source_time_diff, account_age, parent_time_dff,
                avg_child_time_diff, is_verified, num_friends, num_followers,
                num_hastags, num_mentions, vader_score
            ])

            # Extracting features for retweets
            for retweet in self.all_retweets[tweet['id_str']]:

                id = retweet['id_str']

                # Temporal Features
                source_time_diff = retweet['created_at'] - source['publish_date']
                account_age = get_account_age(retweet['created_at'], retweet['user']['created_at'])
                parent_time_dff = get_parent_time_diff(self.graph, self.news_id, retweet['id_str'], retweet['created_at'])
                avg_child_time_diff = get_avg_child_time_diff(self.graph, retweet['id_str'], retweet['created_at'])

                # User-based features
                is_verified = 1 if retweet['user']['verified'] else 0
                num_friends = retweet['user']['friends_count']
                num_followers = retweet['user']['followers_count']
                
                # Text-based features
                num_hastags = len(retweet['entities']['hashtags'])
                num_mentions = len(retweet['entities']['user_mentions'])
                vader_score = get_vader_score(retweet['text'])

                data.append([
                    id, source_time_diff, account_age, parent_time_dff,
                    avg_child_time_diff, is_verified, num_friends, num_followers,
                    num_hastags, num_mentions, vader_score
                ])

        return pd.DataFrame(data=data, columns=feature_names)


    def collect_all_tweets(self) -> list:
        '''
        Returns a list of all available tweets for a news article
        '''
        tweets_path = os.path.join(self.news_path, 'tweets')
        if self.tweets is None:
            tweets = []
            if os.path.exists(tweets_path):
                for tweet_file in os.listdir(tweets_path):
                    tweet = json.load(open(os.path.join(tweets_path, tweet_file)))
                    tweet['created_at'] = twittertime_to_timestamp(tweet['created_at'])
                    tweets.append(tweet)
            self.tweets = tweets
            self.num_of_tweets = len(tweets)
        return self.tweets

    def collect_all_retweets(self, tweet_id) -> list:
        '''
        Returns a list of all available retweets for a tweet
        '''
        if self.all_retweets is None:
            self.all_retweets = json.load(open(os.path.join(self.news_path, 'retweets.json')))
        retweets = self.all_retweets.get(tweet_id, [])
        for i in retweets:
            i.update({'created_at': twittertime_to_timestamp(i['created_at'])})
        return retweets

