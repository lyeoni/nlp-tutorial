import argparse
from bs4 import BeautifulSoup
import pandas as pd

def argparser():
    p = argparse.ArgumentParser()
       
    p.add_argument('--filename',
                    default='Posts.xml')

    config = p.parse_args()
    
    return config

def to_dataframe(input):
    contents = open(input,'r', encoding='utf-8').read()
    soup = BeautifulSoup(contents, 'lxml')
    
    records={}
    use_columns = ['title', 'body', 'tags', 'posttypeid', 'viewcount']
    for ri, row in enumerate(soup.find_all('row')):
        k = int(row.attrs.get('id'))
        v = []
        for col in use_columns:
            v.append(row.attrs.get(col))
        
        records[k] = v # {id : title, body, tags, posttypeid, viewcount}
    
    df = pd.DataFrame.from_dict(records, orient='index', columns=use_columns)
    
    return df

if __name__=='__main__':
    config = argparser()
        
    data = to_dataframe('data/'+config.filename)