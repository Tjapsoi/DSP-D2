import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import ast

file_path = 'all_annotations.xlsx'
df = pd.read_excel(file_path)

la = pd.read_csv('annotations_lena.csv')['LENA SRL TAGS'].iloc[:100]
la[99] = "['O']"

df = pd.concat([df, la], axis=1)

df.rename({
    'LENA SRL TAGS': 'HA 2'
}, inplace=True, axis=1)

models = df.columns

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=1.5,  
                annot_kws={"size": 18}, xticklabels=['ACTOR', 'V', 'OBJ', 'REC', 'O'], yticklabels=['ACTOR', 'V', 'OBJ', 'REC', 'O'])
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0) 
    plt.show()

for model in models:
    print(model)
    unequal = 0

    gt = df['GT'].apply(ast.literal_eval)
    model_ann = df[model].apply(ast.literal_eval)

    for i in range(100):
        if len(gt[i]) != len(model_ann[i]):
            gt.drop(i, axis=0, inplace=True)
            model_ann.drop(i, axis=0, inplace=True)
            unequal += 1

    ground_truth = [str(item) for sublist in gt for item in sublist]
    predicted_labels = [str(item) for sublist in model_ann for item in sublist]

    cm = confusion_matrix(ground_truth, predicted_labels, labels=['ACTOR', 'V', 'OBJ', 'REC', 'O'])
    print(f'Confusion Matrix for {model}:')
    print(cm)
    print(unequal, 'sentences were not the same length as the ground truth')
    plot_confusion_matrix(cm, model)
