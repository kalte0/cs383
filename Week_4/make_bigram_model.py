from collections import defaultdict
import nltk 
import pickle 
from nltk.probability import FreqDist
from nltk.corpus import brown

nltk.download('brown')

text = brown.words()
text = [word.lower() for word in text]

freq_dist = FreqDist(text) 
total_words = len(text) 
prob_dist = {word: freq_dist[word]/total_words for word in freq_dist.keys()}

for word, prob in list(prob_dist.items())[:10]:
    print(f"Probability of '{word}': {prob:.4f}")

def nested_default_float(): 
    return defaultdict(float)

def nested_default_int(): 
    return defaultdict(int)

# Given a list of bigrams appearing in a specific text.
def build_bigram_model(words):
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    
    # Create a defaultdict to store the bigram frequencies
    bigram_freq = defaultdict(nested_default_int)

    # Count occurrences of each bigram
    for word1, word2 in bigrams:
        bigram_freq[word1][word2] += 1

    # Calculate probabilities for each bigram
    bigram_model = defaultdict(nested_default_float)
    for word1 in bigram_freq:
        total_count = sum(bigram_freq[word1].values())
        for word2 in bigram_freq[word1]:
            bigram_model[word1][word2] = bigram_freq[word1][word2] / total_count


    return bigram_model

bigram_model = build_bigram_model(text)

with open('bigram_model.pkl', 'wb') as f: 
  pickle.dump(bigram_model, f) 