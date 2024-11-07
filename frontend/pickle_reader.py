import pickle
import pandas as pd
# Load the pickle file
with open('random_forest_model.pkl', 'rb') as file:
    content = pickle.load(file)

# Print the type of the loaded object
print(f'Type of the content: {type(content)}')

# Print the content based on its type
if isinstance(content, pd.DataFrame):
    # If it's a DataFrame, print the first few rows
    print(content.head())
elif isinstance(content, dict):
    # If it's a dictionary, print the keys
    print(content.keys())
else:
    # If it's another type, just print it
    print(content)
