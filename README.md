# Data Systems Project (Group D2)

This repository contains the code, data, and models used for the **Data Systems Project** by Group D2.

## Team Members
- Bilal Boulbarss  
- Lena Zubik  
- Zubin Pengel  
- Jimmy Tebben  
- Nikolay Filipov  

## Repository Structure
- `data/` – Contains ground truth data and annotations from our experiments.  
  - `all_positives_with_tags_full.csv` – The ground truth dataset.  
- `models/` – Includes the models used in our experiments via Hugging Face.  
- `demo.py` – A demo script that picks a random sentence from the ground truth Dutch constitution (as long as the sentence has less than 20 tokens). The user annotates the sentence. Accuracy is evaluated against the ground truth, then compared (at token level) to the annotations produced by three of the five LLMs used in our experiment, and the two human annotators. 
