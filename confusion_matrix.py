import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import string
import ast

# Load the Excel file
file_path = 'data.csv'
df = pd.read_csv(file_path)

# Extract the ground truth and model annotations
ground_truth = [str(item) for sublist in df['GT'].apply(ast.literal_eval) for item in sublist]
ground_truth = [item for item in ground_truth if not (isinstance(item, (int, float)) or (isinstance(item, str) and item in string.punctuation))]

# Define the models (assuming first three columns are models)
models = df.columns

# Function to plot confusion matrix with grayscale and formatted numbers
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', linewidths=1.5,  
                annot_kws={"size": 18})
    plt.xlabel('Predicted Labels', fontsize=18)
    plt.ylabel('True Labels', fontsize=18)
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18, rotation=0)  # Keep labels upright
    plt.show()

# Generate and plot confusion matrices for each model
for model in models:
    print(model)
    predicted_labels = [str(item) for sublist in df[model].apply(ast.literal_eval) for item in sublist]
    predicted_labels = [item for item in predicted_labels if not (isinstance(item, (int, float)) or (isinstance(item, str) and item in string.punctuation))]
    
    cm = confusion_matrix(ground_truth, predicted_labels, labels=['ACTOR', 'V', 'OBJ', 'REC', 'O'])
    print(f'Confusion Matrix for {model}:')
    print(cm)
    plot_confusion_matrix(cm, model)
