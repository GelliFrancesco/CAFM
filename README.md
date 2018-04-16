# CAFM

Code for Context Aware Factorization Machine (CAFM), as described in ACM MM17 paper: [_How Personality Affects our Likes: Towards a Better Understanding of Actionable Images_ ](https://dl.acm.org/citation.cfm?doid=3123266.3127909)

CAFM can run with any of:
* rating information and context features (as in [this paper](https://dl.acm.org/citation.cfm?id=2964291)) - sparse
* user features (e.g. demographics, personality traits) - dense
* item features (e.g. sentiment, concepts) - dense

For data size limitation, input data is not provided in this page. See instruction below for how to download the data and run the code.


## REQUIREMENTS
Required Python packages: tensorflow, sklearn, numpy, h5py
Note: computation on cpu can be slow.


## DOWNLOAD DATA
Download the data input [here](https://www.dropbox.com/s/e7rbycfvkanq6k6/data.zip?dl=0)
Unpack the folder data/ in the project folder (same level as CAFM.py).
data/ can be used either to replicate the experiments in the paper, or to discover the input data format for new inputs.


## RUNNING THE CODE
After installing the required Python packages and downloading the input data, run the code with:

python CAFM.py (ratings&context only)
or
python CAFM.py --user_ft 1 (ratings&context and user personality traits)

Additional parameters (batch size, learning rate, etc.) can be listed running: python CAFM.py --help


## INCLUDING ITEM FEATURES
Due to file size, image dense features (distribution over sentiment visual concepts) are here not included.
In order to run the code with image sentiment features, please create and include the two files in the project folder:

data/training/item_dense.h5
data/testing/item_dense.h5

Each of these need to be a hdf5 file with a single dataset named "output". Such dataset is a floating-point multidimensional array of size (num_instances, feature_dimensionality). 
In case of sentiment features, feature_dimensionality is 4342 if the English Visual Sentiment Ontology is used.
The oder of the instances should match the order in image_index.txt

In  order to replicate the experiments in the paper, please crawl the raw image files with Twitter API (list provided in data/training/image_index.txt and data/testing/image_index.txt), extract the sentiment features using the English concept detector (caffe model can be download here: mvso.cs.columbia.edu) and create a hdf5 file as explained above.

It can happen that some of the image tweets cannot be crawled (e.g. user was deleted or images were removed). In that case, the missing tweets must be removed from the ratings&context files as well. For that, follow these instructions below.

### HOW TO UPDATE CONTEXT DATASET
In case some image tweets are missing you may need to update the ratings&context dataset. The ratings&context dataset is a sparse representation and the libFM format is used.
If image tweet TWEET_ID is missing, please search each line with TWEET_ID in data/context/index_tr.csv and data/context/index_ts.csv. Each of these lines correspond to a rating (either positive of negative sample) that a user did on a missing image tweet. Such lines must be eliminated both from the .csv and the corresponding .libfm file in the same folder. If such lines are not removed, CAFM.py --item_ft 1 will search for the item features for such missing images and will crash because not able to find them. 

You can finally run the code with:
python CAFM.py --item_ft 1
