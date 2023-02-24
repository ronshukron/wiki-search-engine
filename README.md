# search-engine-for-English-Wikipedia
in this project we have 3 main files and we going to explain here the flow of the code and to give some explanations about the each file,
the flow of the whole project:
in this project we build search engine for English Wikipedia from 2021.

gcp - in this file we build 2 indexes article body index and article title index, we will also build a few important dictinaries, page view dictionary, page rank dictionary and DL dictionay that holds the len of each document length for each index. 
in the notebook you'll find indexs that can be stemmed and are with an upgraded tokenizer.
you can also make a non stemmed index by using the regular word_count function.
we used the pyspark framework to run on a gcp dataproc cluster.

google storage - we save those indexes and dictionary to a few different bucket, we save each index to a different bucket.

google instance - we created an instace server and downloaded the indexes and dictionary to it, save localy, then we run the search_frontend.py file
which creates a flask server ready to receive query and give back relevant wiki articles with an 0.53 map40 score.

search_frontend - in the search fundction you'll find the main information retrival function we used which combines the body and title search results.
you can also use the search_title and search_body functions to get search results from one index.
you can use test function to test you map40 score for each index by it self or combines together.

