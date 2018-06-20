#!/bin/sh

# Installing NLTK Data
# NLTK comes with many corpora, toy grammars, trained models, etc.
# A complete list is posted at: http://nltk.org/nltk_data/ .
# To install the data, first install NLTK (see http://nltk.org/install.html),
# then use NLTK’s data downloader as described below.

# option: -d /usr/local/share/nltk_data => the location to store nltk_data
python -m nltk.downloader -d /usr/local/share/nltk_data all
