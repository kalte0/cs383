import sys
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_text(model, tokenizer, max_sequence_len, starting_word, num_words=50):
    generated_text = starting_word
    for _ in range(num_words):
        encoded = tokenizer.texts_to_sequences([generated_text])[0]
        encoded = pad_sequences([encoded], maxlen=max_sequence_len, padding='pre')
        predicted_index = model.predict_classes(encoded, verbose=0)
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_word = word
                break
        generated_text += " " + predicted_word
    return generated_text

def main():
    if len(sys.argv) != 3:
        print("Usage: python predict_sentence.py <model_filepath> <starting_word>")
        sys.exit(1)

    model_filepath = sys.argv[1]
    starting_word = sys.argv[2]

    # Load pickled model
    try:
        with open(model_filepath, 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Model file '{model_filepath}' not found.")
        sys.exit(1)

    # Generate sentence
    tokenizer = model.tokenizer # Dot accessor for tokenizer
    max_sequence_len = model.max_sequence_len
    generated_sentence = generate_text(model['model'], tokenizer, max_sequence_len, starting_word)
    print("Generated Sentence:", generated_sentence)

if __name__ == "__main__":
    main()
