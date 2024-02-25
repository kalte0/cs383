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

def build_trigram_model(words):
    trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
    # Create a defaultdict to store the trigram frequencies
    trigram_freq = defaultdict(nested_default_int)

    # Count occurrences of each trigram
    for word1, word2, word3 in trigrams:
        trigram_freq[(word1, word2)][word3] += 1

    # Calculate probabilities for each trigram
    trigram_model = defaultdict(nested_default_float)
    for pair in trigram_freq:
        total_count = sum(trigram_freq[pair].values())
        for word3 in trigram_freq[pair]:
            trigram_model[pair][word3] = trigram_freq[pair][word3] / total_count

    return trigram_model

trigram_model = build_trigram_model(text)

with open('trigram_model.pkl', 'wb') as f: 
  pickle.dump(trigram_model, f) 