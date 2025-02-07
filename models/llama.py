"""
guide used: https://huggingface.co/docs/transformers/conversations

keep in mind terms and conditions of llama!

## Llama
"""

from huggingface_hub import login
login(token="your_token")

import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a classifier for tokenized sentences. Your job is to assign a label to each token based on its role in the sentence. Use labels like ACTOR, V, OBJ, REC, O, or punctuation marks. Always provide the labels for each tokenized list you are given."},
    {"role": "user", "content":
     """
You are a classifier tasked with labeling tokenized sentences written in Dutch. Each token in the list corresponds to a word or punctuation mark from a sentence. Your job is to assign the correct label to each token based on its role in the sentence.

Instructions:
Each token must be classified into one of the following labels:

ACTOR: Refers to a subject or entity performing an action.
V: Refers to verbs or actions.
OBJ: Refers to objects or entities receiving an action.
REC: Refers to recipients or beneficiaries.
O: Refers to other words or tokens that don't fit into the above categories.
Punctuation marks and special characters: Retain their original form as labels (e.g., ".", ",", "-").

Important Rules:
- The classification list must have the same length as the tokenized list.
- Each token in the input list must have exactly one corresponding label.
- If the lengths do not match, your output is incorrect. Review and fix it before finalizing.
- The labels must be in the same order as the tokens in the list.

The sentences and tokens are written in Dutch. Make sure to interpret them in the context of Dutch grammar and syntax

Use the examples below as a reference for how to classify tokens. After the examples, classify the tokens in the provided lists.

Examples:
###
List 1: ['Aan', 'de', 'bewoner', 'wordt', 'zo', 'spoedig', 'mogelijk', 'een', 'schriftelijk', 'verslag', 'van', 'het', 'binnentreden', 'verstrekt', '.']
Labels 1: ['O', 'ACTOR', 'ACTOR', 'V', 'O', 'O', 'O', 'OBJ', 'OBJ', 'OBJ', 'OBJ', 'OBJ', 'OBJ', 'V', '.']

List 2: ['De', 'ministers', 'en', 'de', 'staatssecretarissen', 'geven', 'de', 'kamers', 'elk', 'afzonderlijk', 'en', 'in', 'verenigde', 'vergadering', 'mondeling', 'of', 'schriftelijk', 'de', 'door', 'een', 'of', 'meer', 'leden', 'verlangde', 'inlichtingen', 'waarvan', 'het', 'verstrekken', 'niet', 'in', 'strijd', 'is', 'met', 'het', 'belang', 'van', 'de', 'staat', '.']
Labels 2: ['ACTOR', 'ACTOR', 'O', 'ACTOR', 'ACTOR', 'V', 'REC', 'REC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'OBJ', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', '.']

List 3: [‘een', 'ondernemer', 'aan', 'wie', 'een', 'concessieopdracht', 'is', 'gegund']
Labels 3: [‘REC', 'REC', 'O', 'O', 'ACTOR', 'ACTOR', 'V', 'V']
###

Classify the following lists:
"""
}]

pipe = pipeline("text-generation", "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

response = pipe(chat, max_new_tokens=2048)
print(response[0]['generated_text'][-1]['content'])
