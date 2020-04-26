# word_graph2020

Diachronic changes of language include changes in all aspects of language, in which changes in lexical meaning are comparatively fast than other aspects, such like phonological variations and syntactic structures. Lexical meaning evolution always endures a complicate path, therefore in this project. Based on the data we have collected, that is, 400 English literature books between 19th and 20th century (200 for each century) and check how the word meanings change through time in two perspectives.

1. Train word embeddings with lemmatized words and create networks for each time period with word embeddings, then check the target words by doing network analysis. Target words, such like 'awful', 'pretty', etc. (We would collect words from lexical semantics papers)

2. Train word embeddings with word_pos, and create networks, check the role that part-of-speech plays in word meaning evolution, that is, when a word has two or more part-of-speech, the word functions differently in a sentence, its syntactic and semantic context would be different as well. We would like to see how a word's neighbors are different when it has multiple part-of-speech.


## Research Question

(1) To find the words that have meaning changes through time, that is, we compare 17th, 19th and 20th century English literatures.

(2) In order to get good result for (1), we need to figure out the best representation of word, i.e., word embeddings.


## Data

17th, 19th and 20th English literatures downloaded from Gutenberg Project. 

Google News Embeddings

## Method

1. Word2Vec-CBOW, train word embeddings for each time period

2. Randomly sample 1000 words from the words that appear in all time periods

3. for each word, find its frequency, and calculate the Jaccard Efficient.
4. find threshold for word meaning change

5. select examplars

6. using target words from Jurafsky's paper to see how they behave in our model.




