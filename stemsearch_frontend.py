import re
from collections import Counter

import nltk
import numpy as np
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
nltk.download('stopwords')
import pandas as pd

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


import inverted_index_gcp
import pickle

bodyindex = inverted_index_gcp.InvertedIndex.read_index('stem_body_index/postings_gcp', 'bodyindex')
titleindex = inverted_index_gcp.InvertedIndex.read_index('stem_title_index/postings_gcp', 'titleindex')

# bodyindex = inverted_index_gcp.InvertedIndex.read_index('standart_body_index/postings_gcp', 'bodyindex')
# titleindex = inverted_index_gcp.InvertedIndex.read_index('standart_title_index/postings_gcp', 'titleindex')





# Open the pickle file
pkl_file = open('DL_stem_body/my_DL_body_dict/DL_body_dict.pkl', 'rb')

# Load the pickled object
DL_body = pickle.load(pkl_file)

# Close the file
pkl_file.close()

# Open the pickle file
pkl_file2 = open('DL_stem_title/my_DL_title_dict/DL_title_dict.pkl', 'rb')

# Load the pickled object
DL_title = pickle.load(pkl_file2)

# Close the file
pkl_file2.close()

# page view dict
pkl_file3 = open('pageviews-202108-user.pkl', 'rb')

# Load the pickled object
page_view = pickle.load(pkl_file3)

# Close the file
pkl_file3.close()

# page view dict
pkl_file4 = open('page_rank/my_pageRanke_dict/dictionary_pageRanke_dict.pkl', 'rb')

# Load the pickled object
page_rank = pickle.load(pkl_file4)

# Close the file
pkl_file4.close()

# wiki to title dict
pkl_file5 = open('wiki_id_to_title_dict/my_id_and_title_dict/id_and_title_dict.pkl', 'rb')

# Load the pickled object
wiki_id_to_title = pickle.load(pkl_file5)

# Close the file
pkl_file5.close()

# print(len(wiki_id_to_title))
# count = 0
# for i in wiki_id_to_title.items():
#     if count > 100:
#         break
#     count += 1
#     print(i)

def merge_results_title_body_anchor(title_scores, body_scores,title_weight=0.5,text_weight=0.5,N = 3):
  """
    This function merge and sort documents retrieved by its weighte score (e.g., title ,body and anchor).
     Parameters:
     3 lists of docs and scores that way:
     1. list of docs and score : title_scores = [(1,4.5),(2,7)] thats mean doc with id=1 have score 4.5
  """


  dict_docId_score= {}
  # going throw title list:
  for doc in title_scores:
      dict_docId_score[doc[0]] = (doc[1]*title_weight)
  # going throw body list:
  for doc in body_scores:
    if doc[0] in dict_docId_score.keys():
      dict_docId_score[doc[0]] = dict_docId_score[doc[0]] + doc[1]*text_weight
    else:
      dict_docId_score[doc[0]] = doc[1]*text_weight

    tuples_list = list(dict_docId_score.items())
    tuples_list = sorted(tuples_list, key=lambda x: x[1], reverse=True)[:N]
  return  tuples_list


# All the function we need for search_body, search_title and search_anchor:

# tokenize
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
def tokenize(text):
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens
# tokenize

def query_text_to_dict(text):
  tokens = tokenize(text)
  dic = {1: tokens}
  return dic


from nltk.stem.porter import *
stemmer = PorterStemmer()

RE_WORD_advanced = re.compile(r"""[\#\@\w](['\-]?[\w,]?[\w.]?(?:['\-]?[\w,]?[\w])){0,24}""", re.UNICODE)
def tokenize_advanced(text):
    list_of_tokens = [token.group() for token in RE_WORD_advanced.finditer(text.lower()) if token.group() not in all_stopwords]
    list_of_tokens = [stemmer.stem(token) for token in list_of_tokens]
    return list_of_tokens

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer

def query_text_to_dict_advanced(text):
  tokens = tokenize_advanced(text)
  dic = {1: tokens}
  return dic


def cosine_similarity(query,index, index_name,DL, N=3, withN=True):

  similarity_scores = {}
  nq = 1 / len(query.values())

  for terms in query.values():
      for term in terms:
          posting_list = index.read_posting_list(index, term, index_name)
          for doc_id, tf in posting_list:
             similarity_scores[doc_id] = 0

  for terms in query.values():
      counter = Counter(terms)
      for term in terms:
         posting_list = index.read_posting_list(index, term, index_name)
         for doc_id,doc_tf in posting_list:
           similarity_scores[doc_id] = similarity_scores.get(doc_id, 0) + counter[term]*doc_tf

  for doc_id, tf in similarity_scores.items():
      similarity_scores[doc_id] = similarity_scores.get(doc_id, 0)*nq*(1/(DL[str(doc_id)]))
      # similarity_scores[doc_id] = similarity_scores.get(doc_id, 0)*nq*1

  # print(similarity_scores)
  if withN==True:
    return sorted([(doc_id, round(score, 5)) for doc_id, score in similarity_scores.items()], key=lambda x: x[1],reverse=True)[:N]
  else:
    return sorted([(doc_id, round(score, 5)) for doc_id, score in similarity_scores.items()], key = lambda x: x[1], reverse=True)

def cosine_similarity_for_body(query,index, index_name,DL, N=3, withN=True):

  similarity_scores = {}
  #######################
  nq = 1 / len(query.values())
  #######################
  for terms in query.values():
      for term in terms:
          posting_list = index.read_posting_list(index, term, index_name)
          for doc_id, tf in posting_list:
             similarity_scores[doc_id] = 0

  for terms in query.values():
      counter = Counter(terms)
      for term in terms:
         posting_list = index.read_posting_list(index, term, index_name)
         for doc_id,doc_tf in posting_list:
           similarity_scores[doc_id] = similarity_scores.get(doc_id, 0) + counter[term]*doc_tf

  for doc_id, tf in similarity_scores.items():
      similarity_scores[doc_id] = similarity_scores.get(doc_id, 0)*nq*(1/(DL[str(doc_id)]))
      # similarity_scores[doc_id] = similarity_scores.get(doc_id, 0)*nq*1

  if withN==True:
    return sorted([(doc_id, round(score, 5)) for doc_id, score in similarity_scores.items()], key=lambda x: x[1],reverse=True)[:N]
  else:
    return sorted([(doc_id, round(score, 5)) for doc_id, score in similarity_scores.items()], key = lambda x: x[1], reverse=True)


def get_page_rank(doc_id_list, N=3):
    # get doc_id_list = [(wiki_id, title)...]
    res = []
    for doc in doc_id_list:
        # return [(page_rank,wiki_id, title)]
        res.append((page_rank[doc[0]], doc[0], doc[1]))
    sort_list = sorted(res, key=lambda x: x[0], reverse=True)[:N]
    result = []
    for i in sort_list:
        result.append((i[1], i[2]))

    return result

def get_page_view(doc_id_list, N=3):
    # get doc_id_list = [(wiki_id, title)...]
    res = []
    for doc in doc_id_list:
        # return [(page_rank,wiki_id, title)]
        res.append((page_view[doc[0]], doc[0], doc[1]))
    sort_list = sorted(res, key=lambda x: x[0], reverse=True)[:N]
    result = []
    for i in sort_list:
        result.append((i[1], i[2]))

    return result

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query = query_text_to_dict_advanced(query)
    if len(query) == 1:
        title = cosine_similarity(query, titleindex, "stem_title_index", DL_title, 100, True)
        res = title

    elif len(query) == 2:
        body = cosine_similarity_for_body(query, bodyindex, "stem_body_index", DL_body, 500, True)
        title = cosine_similarity(query, titleindex, "stem_title_index", DL_title, 100, False)
        res = merge_results_title_body_anchor(title, body, 0.9, 0.1, 300)
    else:
        body = cosine_similarity_for_body(query, bodyindex, "stem_body_index", DL_body, 500, True)
        title = cosine_similarity(query, titleindex, "stem_title_index", DL_title, 100, False)
        res = merge_results_title_body_anchor(title, body, 0.3, 0.7, 300)

    result = []
    for i in range(len(res)):
        result.append((res[i][0], wiki_id_to_title[res[i][0]]))

    # result = get_page_rank(result, 200)
    # result = get_page_view(result, 100)
    # END SOLUTION
    return jsonify(result)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      res.append(1111111111111)
      return jsonify(res)
    # BEGIN SOLUTION

    # tokenizer the query:

    # q_token = Tokenizer(query)
    # if we send True to get_topN_score_for_queries its mean that it will returns 100 search results
    d = cosine_similarity_for_body(query_text_to_dict_advanced(query), bodyindex, "stem_body_index", DL_body, 100, True)
    res_temp = d
    res = res_temp
    result = []
    for i in range(len(res)):
        result.append((res[i][0], wiki_id_to_title[res[i][0]]))

    # res = []
    # get_wiki_title a dict that returns the title of the wiki_id that he get
    # for i in res_temp:
    #   res.append((i[0],get_wiki_title(i[0])))

    # END SOLUTION
    return jsonify(result)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      res.append(1111111111111)
      return jsonify(res)
    # BEGIN SOLUTION
    # if we send False to get_topN_score_for_queries its mean that it will returns all the search results
    d = cosine_similarity(query_text_to_dict_advanced(query), titleindex, "stem_title_index", DL_title, 100, True)
    res_temp = d
    res = res_temp

    result = []
    for i in range(len(res)):
        result.append((res[i][0], wiki_id_to_title[res[i][0]]))

    # END SOLUTION
    return jsonify(result)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    # if we send False to get_topN_score_for_queries its mean that it will returns all the search results
    # d = get_topN_score_for_queries(query_text_to_dict(query),idx_anchor,100,False)
    # res_temp = d[1]
    # res = []
    # # get_wiki_title a dict that returns the title of the wiki_id that he get
    # for i in res_temp:
    #     res.append((i[0], get_wiki_title(i[0])))

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    res = []
    for i in range(len(wiki_ids)):
        res.append((wiki_ids[i], page_rank[wiki_ids[i]]))
    print(res)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = []
    # print(type(page_rank))
    for i in range(len(wiki_ids)):
        res.append((wiki_ids[i], page_view[wiki_ids[i]]))
    # print(res)
    # END SOLUTION
    return jsonify(res)



@app.route("/test_body")
def test_body():
    import json

    with open('queries_train.json', 'rt') as f:
        queries = json.load(f)

    def average_precision(true_list, predicted_list, k=40):
        true_set = frozenset(true_list)
        predicted_list = predicted_list[:k]
        precisions = []
        for i, doc_id in enumerate(predicted_list):
            if doc_id in true_set:
                prec = (len(precisions) + 1) / (i + 1)
                precisions.append(prec)
        if len(precisions) == 0:
            return 0.0
        return round(sum(precisions) / len(precisions), 3)

    import requests
    from time import time
    # url = 'http://35.232.59.3:8080'
    # place the domain you got from ngrok or GCP IP below.
    url = 'http://35.238.224.210:8080'
    map_score = 0
    my_dur = 0
    qs_res = []
    for q, true_wids in queries.items():
        duration, ap = None, None
        t_start = time()
        try:
            res = requests.get(url + '/search_body', {'query': q}, timeout=35)
            duration = time() - t_start
            my_dur += duration
            if res.status_code == 200:
                pred_wids, _ = zip(*res.json())
                ap = average_precision(true_wids, pred_wids)
                map_score += ap

        except:
            pass

        qs_res.append((q, duration, ap))

    print(map_score/30)
    print(my_dur/30)

    # END SOLUTION
    return jsonify(qs_res)

@app.route("/test_title")
def test_title():
    import json

    with open('queries_train.json', 'rt') as f:
        queries = json.load(f)

    def average_precision(true_list, predicted_list, k=40):
        true_set = frozenset(true_list)
        predicted_list = predicted_list[:k]
        precisions = []
        for i, doc_id in enumerate(predicted_list):
            if doc_id in true_set:
                prec = (len(precisions) + 1) / (i + 1)
                precisions.append(prec)
        if len(precisions) == 0:
            return 0.0
        return round(sum(precisions) / len(precisions), 3)

    import requests
    from time import time
    # url = 'http://35.232.59.3:8080'
    # place the domain you got from ngrok or GCP IP below.
    url = 'http://35.238.224.210:8080'
    map_score = 0
    my_dur = 0

    qs_res = []
    for q, true_wids in queries.items():
        duration, ap = None, None
        t_start = time()
        try:
            res = requests.get(url + '/search_title', {'query': q}, timeout=35)
            duration = time() - t_start
            my_dur += duration

            if res.status_code == 200:
                pred_wids, _ = zip(*res.json())
                ap = average_precision(true_wids, pred_wids)
                map_score += ap
        except:
            pass

        qs_res.append((q, duration, ap))

    print(map_score/30)
    print(my_dur/30)


    # END SOLUTION
    return jsonify(qs_res)

@app.route("/test")
def test():
    import json

    with open('queries_train.json', 'rt') as f:
        queries = json.load(f)

    def average_precision(true_list, predicted_list, k=40):
        true_set = frozenset(true_list)
        predicted_list = predicted_list[:k]
        precisions = []
        for i, doc_id in enumerate(predicted_list):
            if doc_id in true_set:
                prec = (len(precisions) + 1) / (i + 1)
                precisions.append(prec)
        if len(precisions) == 0:
            return 0.0
        return round(sum(precisions) / len(precisions), 3)

    import requests
    from time import time
    # url = 'http://35.232.59.3:8080'
    # place the domain you got from ngrok or GCP IP below.
    url = 'http://35.238.224.210:8080'
    map_score = 0
    my_dur = 0

    qs_res = []
    for q, true_wids in queries.items():
        duration, ap = None, None
        t_start = time()
        try:
            res = requests.get(url + '/search', {'query': q}, timeout=35)
            duration = time() - t_start
            my_dur += duration

            if res.status_code == 200:
                pred_wids, _ = zip(*res.json())
                ap = average_precision(true_wids, pred_wids)
                map_score += ap
        except:
            pass

        qs_res.append((q, duration, ap))

    print(map_score/30)
    print(my_dur/30)

    # END SOLUTION
    return jsonify(qs_res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)





