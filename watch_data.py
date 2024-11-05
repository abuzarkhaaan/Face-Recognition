import pickle
import numpy as np

# Load the known embeddings from the .pkl file
try:
    with open("embedding_database.pkl", "rb") as f:
        embedding_database = pickle.load(f)

    if not isinstance(embedding_database, dict):
        print("The data in the file is not in the expected format.")
    else:
        # Print the names and a snippet of the embeddings
        for name, embedding in embedding_database.items():
            print(f"Name: {name}")
            print(f"Embedding snippet: {embedding[:5]}... (total length: {len(embedding)})")
            print("\n")

except FileNotFoundError:
    print("The embedding database file does not exist.")
