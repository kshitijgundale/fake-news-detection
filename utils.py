from datetime import datetime
import networkx as nx

def twittertime_to_timestamp(twitter_time) -> float:
    '''
    Converts twitter time to POSIX timestamp
    '''
    return datetime.strptime(twitter_time, '%a %b %d %H:%M:%S +0000 %Y').timestamp()

def get_account_age(tweet_created_at, user_created_at) -> float:
    '''
    Returns account age at the time of tweet creation
    '''
    return tweet_created_at - twittertime_to_timestamp(user_created_at)

def get_parent_time_diff(g: nx.DiGraph, news_id, tweet_id, created_at) -> float:
    '''
    Returns difference in creation time of parent node and child node
    '''
    parent_id = g.predecessors(tweet_id)[0]
    if parent_id == news_id:
        return created_at - g.nodes[parent_id]['created_at']
    return 0

def get_avg_child_time_diff(g: nx.DiGraph, tweet_id, created_at):
    '''
    Returns average time difference with immediate child nodes
    '''
    child_ids = g.successors(tweet_id)
    if len(child_ids) != 0:
        return [g.nodes[x]['created_at'] - created_at for x in child_ids]
    return 0

def get_vader_score(text, vader):
    '''
    Returns VADER sentiment score for given text
    '''
    return vader.polarity_scores(text)['compound']