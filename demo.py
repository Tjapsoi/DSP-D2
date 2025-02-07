import pandas as pd
import ast
import random
import os

os.system('clear')

r_index = random.choice(range(0, 100))

text = pd.read_csv('all_positives_with_tags_full.csv', sep=';').Brontekst.iloc[:100]
tags = pd.read_csv('all_positives_with_tags_full.csv', sep=';').srl_tags.iloc[:100]
tokens = pd.read_csv('all_positives_with_tags_full.csv', sep=';').tokens.iloc[:100]

# Turn the strings from the df into lists
for i in range(len(tags)):
    tags[i] = ast.literal_eval(tags[i])

for i in range(len(tokens)):
    tokens[i] = ast.literal_eval(tokens[i])

# Text, tags and tokens into single df
df = pd.concat([tokens, tags, text], axis=1)

# Choose a text that has less than 20 tokens
while len(df.tokens[r_index]) > 20:
    r_index = random.choice(range(0, 100))

# Seperate all the tokens
tokens_formatted = ''
for token in df.tokens[r_index]:
    tokens_formatted += token + '\n'

complete_text = ''
for token in df.tokens[r_index]:
    complete_text += f'{token} '

# Main instructions
print('Here is your text to annotate:\n\n' +
      complete_text +
      '\n\nHere is the tokenized version:\n\n' +
      tokens_formatted + 
      '\n\nNow its your turn! Annotate the tokens with the following labels: ACTOR, OBJ (object), REC (recipient), V (verb), O (nothing). Numbers and special characters stay the same.\nGood luck!\n')

# Let the user input his tokens
user_tokens = []
for token in df.tokens[r_index]:
    label = input(token + ' -- ')
    user_tokens.append(label)

# Compare user input with our annotations
la = pd.read_csv('annotations_lena.csv')['LENA SRL TAGS'].iloc[:100]              # Lena Annotation 
la[99] = "['O']"
ja = pd.read_excel('annotations_jimmy_verb.xlsx').tags_Jimmy                      # Jimmy Annotation 
o1 = pd.read_excel('annotations_o1.xlsx').srl_tags.iloc[:100]                     # o1 Annotation
llama = pd.read_csv('annotations_llama.csv', sep=';').srl_tags                    # Llama annotation
claude = pd.read_excel('annotations_claude.xlsx').srl_tags                        # Claude annotation

# Turn our annotations into lists
for i in range(100):
    la[i] = ast.literal_eval(la[i])
    ja[i] = ast.literal_eval(ja[i])
    o1[i] = ast.literal_eval(o1[i])
    llama[i] = ast.literal_eval(llama[i])
    claude[i] = ast.literal_eval(claude[i])

gt_tokens = df.srl_tags[r_index]
la_tokens = la[r_index]
ja_tokens = ja[r_index]
o1_tokens = o1[r_index]
llama_tokens = llama[r_index]
claude_tokens = claude[r_index]

os.system('clear')

def write_side_by_side(file_name, *lists):
    # Determine the maximum length of the lists
    max_length = max(len(lst) for lst in lists)

    # Normalize the lengths of the lists by padding with empty strings
    normalized_lists = [lst + [''] * (max_length - len(lst)) for lst in lists]

    placeholder_names = ['Token', 'Ground Truth', 'Your input', 'HA 1', 'HA 2', 'GPT o1', 'claude', 'Llama-8b']

    with open(file_name, 'w') as f:
        f.write("| " + " | ".join(placeholder_names) + " |\n")
        f.write("|" + "|".join(["-" * len(name) for name in placeholder_names]) + "|\n")

        for row in zip(*normalized_lists):
            f.write("| " + " | ".join(row) + " |\n")

write_side_by_side("output.md", df.tokens[r_index], gt_tokens, user_tokens, la_tokens, ja_tokens, o1_tokens, claude_tokens, llama_tokens)

nr_correct_gt = len(gt_tokens)

nr_correct_user = 0
nr_correct_la = 0
nr_correct_ja = 0
nr_correct_o1 = 0
nr_correct_claude = 0
nr_correct_llama = 0

for i in range(len(gt_tokens)):
    if len(gt_tokens) == len(user_tokens):
        if gt_tokens[i] == user_tokens[i]:
            nr_correct_user += 1

    if len(gt_tokens) == len(la_tokens):
        if gt_tokens[i] == la_tokens[i]:
            nr_correct_la += 1

    if len(gt_tokens) == len(ja_tokens):
        if gt_tokens[i] == ja_tokens[i]:
            nr_correct_ja += 1
            
    if len(gt_tokens) == len(o1_tokens):
        if gt_tokens[i] == o1_tokens[i]:
            nr_correct_o1 += 1

    if len(gt_tokens) == len(claude_tokens):            
        if gt_tokens[i] == claude_tokens[i]:
            nr_correct_claude += 1

    if len(gt_tokens) == len(llama_tokens):            
        if gt_tokens[i] == llama_tokens[i]:
            nr_correct_llama += 1

acc_user = round(nr_correct_user/nr_correct_gt*100, 1)

print(f'\nYou achieved an accuracy of {acc_user}%!\n')

acc_la = round(nr_correct_la/nr_correct_gt*100, 1)
acc_ja = round(nr_correct_ja/nr_correct_gt*100, 1)
acc_o1 = round(nr_correct_o1/nr_correct_gt*100, 1)
acc_claude = round(nr_correct_claude/nr_correct_gt*100, 1)
acc_llama = round(nr_correct_llama/nr_correct_gt*100, 1)

variables = {
    "human annotator 1": acc_la,
    "human annotator 2": acc_ja,
    "GPT-o1": acc_o1,
    "Llama (small)": acc_llama,
    "Claude": acc_claude
}

equal_to = [name for name, value in variables.items() if acc_user == value]
greater_than = [name for name, value in variables.items() if acc_user > value]
worse_than = [name for name, value in variables.items() if acc_user < value]

equal_to = ['no one'] if equal_to == [] else equal_to
greater_than = ['no one'] if greater_than == [] else greater_than
worse_than = ['no one'] if worse_than == [] else worse_than

equal_to = ", ".join(equal_to)
greater_str = ", ".join(greater_than)
worse_str = ", ".join(worse_than)

print(f"You performed just as good as: {equal_to}, ")
print(f"performed better than: {greater_str},")
print(f"and performed worse than: {worse_str}\n\n")
