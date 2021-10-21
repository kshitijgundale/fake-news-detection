from datetime import datetime

def twittertime_to_timestamp(twitter_time) -> float:
    '''
    Converts twitter time to POSIX timestamp
    '''
    return datetime.strptime(twitter_time, '%a %b %d %H:%M:%S +0000 %Y').timestamp()

def get_account_age(tweet_created_at, user_created_at) -> float:
    '''
    Returns account age at the time of tweet creation
    '''
    return twittertime_to_timestamp()

def get_parent_time_diff() -> float:
    '''
    Returns difference in creation time of parent node and child node
    '''