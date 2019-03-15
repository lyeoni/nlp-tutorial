import re

# Remove emoji from an input string
def clean_emoji(input):
    regex = re.compile('[\U00010000-\U0010ffff][\u20000000-\u2fffffff][\U0001f000-\U0001ffff]', flags=re.UNICODE)
    return re.sub(regex, '', input)

# Remove html tags from an input string
def clean_tag(input):
    regex = re.compile('<.*?>')
    return re.sub(regex, '', input)

# Remove url from an input string
def clean_url(input):
    return re.sub(r'http\S+', '', input)
