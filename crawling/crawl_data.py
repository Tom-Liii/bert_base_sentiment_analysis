from pdb import line_prefix
import requests
import json
import os

def print_review(data, output_file): 
    print('------------------------------')
    print('score: %s' % (data['score']))
    print('review: %s' % (data['review']))
    print(data['review'], file=output_file)
    '''
    if int(data['score']) < 3: 
        print('neg')
        print(data['review'], file=neg_file)
    else: 
        print('pos')
        print(data['review'], file=pos_file)
    '''

def iter_review(no_of_page):
    url = "https://www.rottentomatoes.com/napi/movie/f3a12631-5b97-3a3c-aa65-babf9acebf1a/reviews/user"
    params = {
            "direction" : "next", 
            "endCursor" : "eyJyZWFsbV91c2VySWQiOiJSVF83ODk1Njk4ODQiLCJlbXNJZCI6ImYzYTEyNjMxLTViOTctM2EzYy1hYTY1LWJhYmY5YWNlYmYxYSIsImVtc0lkX2hhc1Jldmlld0lzVmlzaWJsZSI6ImYzYTEyNjMxLTViOTctM2EzYy1hYTY1LWJhYmY5YWNlYmYxYV9UIiwiY3JlYXRlRGF0ZSI6IjIwMjItMDgtMTJUMDM6MjE6NDAuMDY2WiJ9"
    }

    print(url)
    page = requests.get(url, params=params)
    data = json.loads(page.content)
    for review in data['reviews']: 
        yield review
    cursor = data['pageInfo']['endCursor']
    has_next = data['pageInfo']['hasNextPage']

    i = 0
    while(i < no_of_page and has_next): 
        params = {
            "direction" : "next",
            "endCursor" : cursor
        }
        print(cursor)
        page = requests.get(url, params=params)
        data = json.loads(page.content)
        for review in data['reviews']:
            yield review
        cursor = data['pageInfo']['endCursor']
        has_next = data['pageInfo']['hasNextPage']
        i += 1
        if has_next and i < no_of_page: print("===>next page")
        

if __name__ == '__main__':
    if not os.path.exists('./test_set_from_crawling/'):
        os.mkdir('./test_set_from_crawling/')
    with open('./test_set_from_crawling/reviews.txt', 'w', encoding='utf-8') as data_file: 
        print(data_file)
        for data in iter_review(3):
            print_review(data, data_file)
    print('Crawling completed.')
