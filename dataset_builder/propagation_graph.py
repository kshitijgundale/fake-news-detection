import os
import json
import pandas as pd
from utils import *
import networkx as nx
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path

class PropagationGraphBuilder():
    '''
    Each tweet is assumed to be in seperate json file in tweets folder and retweets for all tweets are in single json file named retweets.json
    '''

    def __init__(self, label, news_id, news_path, retweet_gap, vader: SentimentIntensityAnalyzer) -> None:
        self.label = label
        self.news_path = news_path
        self.news_id = news_id
        self.retweet_gap = retweet_gap
        self.vader = vader
        self.all_retweets = None
        self.tweets = None
        self.users = set()
        self.graph = None

    def build_edges(self, save_to_file, save_location=None) -> list:
        '''
        Builds edges of propagation graphs, given tweets and their retweets.
        '''

        # Create networkx graph with tweets/retweets as nodes
        # and keys of tweet/retweet object as node attributes for easy access 
        # when building node features
        g = nx.DiGraph()
        self.graph = g

        # Sorting tweets based on timestamp
        tweets = self.collect_all_tweets()
        if not tweets:
            return
        tweets = sorted(tweets, key=lambda x: x['created_at'])
        self.graph.add_nodes_from([
            (x['id_str'], x) for x in tweets
        ])

        # Build news node
        news_node = json.load(open(os.path.join(self.news_path, 'news_article.json')))
        news_node['created_at'] = tweets[0]['created_at']
        self.graph.add_node(self.news_id, **news_node)

        # Searching immediate source of tweet and updating edges list
        edges = [] # List of tuples -> (ParentNode, ChildNode)

        ## Searching for rest of tweets
        for i in range(0, len(tweets)):
            target_tweet = tweets[i]
            self.users.add(target_tweet['user']['id_str'])
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
            retweets = self.collect_all_retweets(target_tweet['id_str'])
            
            if retweets:
                self.graph.add_nodes_from([
                    (x['id_str'], x) for x in retweets
                ])
                retweets = sorted(retweets, key=lambda x: x['created_at'])
                for i in range(0, len(retweets)):
                    target_retweet = retweets[i]
                    self.users.add(target_retweet['user']['id_str'])
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

        # add edges to graph
        self.graph.add_edges_from(edges)

        # save edges to txt file
        if save_to_file:
            p = Path(save_location)
            p.mkdir(parents=True,exist_ok=True)
            with (p/f"{self.news_id}_graph.txt").open('w') as f:
                json.dump(edges, f)

        return edges


    def build_node_features(self, save_to_file, save_location=None) -> pd.DataFrame:
        '''
        Returns a pandas dataframe of node-level features
        '''

        feature_names = [
            'id',
            'type',
            'created_at',
            'source_time_diff',
            'account_age',
            'parent_time_diff',
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
        source = self.graph.nodes[self.news_id]
        for tweet in tweets:

            id = tweet['id_str']
            type = 'tweet'
            created_at = tweet['created_at']
            
            # Temporal Features
            source_time_diff = tweet['created_at'] - source['created_at']
            account_age = get_account_age(tweet['created_at'], tweet['user']['created_at'])
            parent_time_dff = get_parent_time_diff(self.graph, tweet['id_str'], tweet['created_at'])
            avg_child_time_diff = get_avg_child_time_diff(self.graph, tweet['id_str'], tweet['created_at'])

            # User-based features
            is_verified = 1 if tweet['user']['verified'] else 0
            num_friends = tweet['user']['friends_count']
            num_followers = tweet['user']['followers_count']
            
            # Text-based features
            num_hastags = len(tweet['entities']['hashtags'])
            num_mentions = len(tweet['entities']['user_mentions'])
            vader_score = get_vader_score(tweet['text'], self.vader)

            data.append([
                id, type, created_at, source_time_diff, account_age, parent_time_dff,
                avg_child_time_diff, is_verified, num_friends, num_followers,
                num_hastags, num_mentions, vader_score
            ])

            # Extracting features for retweets
            for retweet in self.all_retweets[tweet['id_str']]:

                id = retweet['id_str']
                type = 'retweet'
                created_at = retweet['created_at']

                # Temporal Features
                source_time_diff = retweet['created_at'] - source['created_at']
                account_age = get_account_age(retweet['created_at'], retweet['user']['created_at'])
                parent_time_dff = get_parent_time_diff(self.graph, retweet['id_str'], retweet['created_at'])
                avg_child_time_diff = get_avg_child_time_diff(self.graph, retweet['id_str'], retweet['created_at'])

                # User-based features
                is_verified = 1 if retweet['user']['verified'] else 0
                num_friends = retweet['user']['friends_count']
                num_followers = retweet['user']['followers_count']
                
                # Text-based features
                num_hastags = len(retweet['entities']['hashtags'])
                num_mentions = len(retweet['entities']['user_mentions'])
                vader_score = get_vader_score(retweet['text'], self.vader)

                data.append([
                    id, type, created_at, source_time_diff, account_age, parent_time_dff,
                    avg_child_time_diff, is_verified, num_friends, num_followers,
                    num_hastags, num_mentions, vader_score
                ])

        df = pd.DataFrame(data=data, columns=feature_names)

        if save_to_file:
            df.to_csv(os.path.join(save_location, f'{self.news_id}_nf.csv'), index=False)

        return df


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
            i['created_at'] = twittertime_to_timestamp(i['created_at'])
        return retweets

    def graph_statistics(self, save_to_file, save_location=None):
        '''
        Returns a pandas dataframe containing metadata about graph
        '''

        num_nodes = len(self.graph.nodes)
        num_tweets = len(self.tweets)
        num_users = len(self.users)
        retweet_perc = (num_nodes - num_tweets)/num_nodes
        total_propagation_time = self.get_total_propagation_time()
        avg_time_diff = self.get_avg_time_diff()

        perc_post_1_hour = 0
        users_10h = set()
        avg_num_followers = {}
        avg_num_friends = {}
        avg_num_retweet = 0

        source = self.tweets[0]
        for tweet in self.tweets:
            if (tweet['created_at'] - source['created_at'])/3600 < 1: 
                perc_post_1_hour += 1
            if (tweet['created_at'] - source['created_at'])/3600 < 10:
                users_10h.add(tweet['user']['id_str'])
            avg_num_retweet += len(self.all_retweets[tweet['id_str']])
            avg_num_followers[tweet['id_str']] = tweet['user']['followers_count']
            avg_num_friends[tweet['id_str']] = tweet['user']['friends_count']

            for retweet in self.all_retweets[tweet['id_str']]:
                if (retweet['created_at'] - source['created_at'])/3600 < 1: 
                    perc_post_1_hour += 1
                elif (retweet['created_at'] - source['created_at'])/3600 < 10:
                    users_10h.add(retweet['user']['id_str'])
                avg_num_followers[retweet['id_str']] = retweet['user']['followers_count']
                avg_num_friends[retweet['id_str']] = retweet['user']['friends_count']

        perc_post_1_hour = perc_post_1_hour/num_nodes
        users_10h = len(users_10h)
        avg_num_retweet = avg_num_retweet/len(self.tweets)
        avg_num_followers = sum(avg_num_followers.values())/len(avg_num_followers)
        avg_num_friends = sum(avg_num_friends.values())/len(avg_num_friends)

        data = [
            ['label', self.label, 'Label'],
            ['num_nodes', num_nodes, 'Total number of nodes in propagation graph'],
            ['num_tweets', num_tweets, 'Total number of tweets in propagation graph'],
            ['avg_num_retweet', avg_num_retweet, 'Average number of retweets per tweet'],
            ['retweet_perc', retweet_perc, 'Percentage of retweets'],
            ['num_users', num_users, 'Number of unique users involved in propagation of news'],
            ['total_propagation_time', total_propagation_time, 'Total propagation time'],
            ['avg_num_followers', avg_num_followers, 'Average of followers counts of users that tweeted or retweeted'],
            ['avg_num_friends', avg_num_friends, 'Average of friends counts of users that tweeted or retweeted'],
            ['avg_time_diff', avg_time_diff, 'Average time between a tweet and acorresponding retweet'],
            ['perc_post_1_hour', perc_post_1_hour, 'Percentage of tweets and retweets posted in first hour'],
            ['users_10h', users_10h, 'Number of unique users who posted in first 10 hours']
        ]

        df = pd.DataFrame(data=data, columns=['feature', 'value', 'description'])

        if save_to_file:
            df.to_csv(os.path.join(save_location, f'{self.news_id}_stats.csv'), index=False)

        return df

    def get_total_propagation_time(self) -> float:
        last_tweet_time = self.tweets[-1]['created_at']
        l = [self.all_retweets[x['id_str']][-1]['created_at'] for x in self.tweets if self.all_retweets[x['id_str']]]
        last_retweet_time = max(l) if l else 0

        return max(last_retweet_time, last_tweet_time)

    def get_avg_time_diff(self) -> float:
        l = [mean(
                [y['created_at'] - x['created_at'] for y in self.all_retweets[x['id_str']]]
            ) for x in self.tweets if self.all_retweets[x['id_str']]]
        return mean(l) if l else 0
