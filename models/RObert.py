from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-ner")
model = AutoModelForTokenClassification.from_pretrained("pdelobelle/robbert-v2-dutch-ner")

# Initialize the NER pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Define the label mapping
label_mapping = {
    "PER": "ACTOR",   # Person -> ACTOR
    "ORG": "ACTOR",   # Organization -> ACTOR
    "LOC": "OBJ",     # Location -> OBJ
    "MISC": "O",      # Miscellaneous -> O
}

# Function to apply custom rules for labeling tokens
def apply_custom_rules(token, ner_label):
    # Example rules to classify verbs, recipients, objects, and punctuation
    if token.istitle() and ner_label == "O":  # Potentially an ACTOR if capitalized and not classified
        return "ACTOR"
    if token in [".", ",", "-", ";", ":"]:    # Punctuation handling
        return token
    if token.endswith("en"):                  # Simple heuristic for Dutch infinitive verbs
        return "V"
    if token in ["aan", "voor", "tegen"]:      # Prepositions that might indicate a recipient
        return "REC"
    return ner_label

# Function to classify tokens using the model and custom rules
def classify_tokens(tokens):
    sentence = " ".join(tokens)  # Join tokens into a sentence for the pipeline
    result = nlp(sentence)

    # Initialize classification with 'O'
    classification = ['O'] * len(tokens)

    # Map predictions to token positions
    for item in result:
        word = item['word']
        label = item['entity_group']
        custom_label = label_mapping.get(label, 'O')

        # Find the matching token index and assign the custom label
        for i, token in enumerate(tokens):
            if word in token and classification[i] == 'O':  # Handle subword tokenization edge cases
                classification[i] = apply_custom_rules(token, custom_label)
                break

    # Apply custom rules to any remaining unclassified tokens
    classification = [apply_custom_rules(tokens[i], classification[i]) for i in range(len(tokens))]

    return classification

# Define tokenized Dutch sentence lists
token_lists = [
    ['verscheidene', 'concurrerende', 'ondernemingen', 'die', 'een', 'dienst', 'mogen', 'verrichten', 'of', 'een', 'activiteit', 'mogen', 'uitoefenen', 'op', 'een', 'andere', 'wijze', 'dan', 'volgens', 'deze', 'criteria', 'worden', 'aangewezen', ',', 'of'],
    ['op', 'een', 'andere', 'wijze', 'dan', 'volgens', 'deze', 'criteria', 'aan', 'een', 'of', 'meer', 'ondernemingen', 'die', 'een', 'dienst', 'mogen', 'verrichten', 'of', 'een', 'activiteit', 'mogen', 'uitoefenen', 'voordelen', 'worden', 'toegekend', ',', 'waardoor', 'enige', 'andere', 'onderneming', 'aanzienlijk', 'wordt', 'belemmerd', 'in', 'de', 'mogelijkheid', 'om', 'dezelfde', 'activiteiten', 'binnen', 'hetzelfde', 'geografische', 'gebied', 'onder', 'in', 'wezen', 'dezelfde', 'omstandigheden', 'uit', 'te', 'oefenen'],
    ['een', 'ondernemer', 'aan', 'wie', 'een', 'concessieopdracht', 'is', 'gegund'],
    ['een', 'schriftelijke', 'overeenkomst', 'onder', 'bezwarende', 'titel', 'die', 'is', 'gesloten', 'tussen', 'een', 'of', 'meer', 'dienstverleners', 'en', 'een', 'of', 'meer', 'aanbestedende', 'diensten', 'of', 'speciale', '-', 'sectorbedrijven', 'en', 'die', 'betrekking', 'heeft', 'op', 'het', 'verlenen', 'van', 'andere', 'diensten', 'dan', 'die', 'welke', 'vallen', 'onder', 'overheidsopdracht', 'voor', 'werken', ',', 'en', 'waarvoor', 'de', 'tegenprestatie', 'bestaat', 'uit', 'hetzij', 'uitsluitend', 'het', 'recht', 'de', 'dienst', 'die', 'het', 'voorwerp', 'van', 'de', 'overeenkomst', 'vormt', ',', 'te', 'exploiteren', ',', 'hetzij', 'uit', 'dit', 'recht', 'en', 'een', 'betaling'],
    ['een', 'schriftelijke', 'overeenkomst', 'onder', 'bezwarende', 'titel', 'die', 'is', 'gesloten', 'tussen', 'een', 'of', 'meer', 'aannemers', 'en', 'een', 'of', 'meer', 'aanbestedende', 'diensten', 'of', 'speciale', '-', 'sectorbedrijven', 'en', 'die', 'betrekking', 'heeft', 'op', ':', 'concessieopdracht', 'voor', 'werken', ':'],
    ['waarvoor', 'de', 'tegenprestatie', 'bestaat', 'uit', 'hetzij', 'uitsluitend', 'het', 'recht', 'het', 'werk', 'dat', 'het', 'voorwerp', 'van', 'de', 'opdracht', 'vormt', ',', 'te', 'exploiteren', ',', 'hetzij', 'uit', 'dit', 'recht', 'en', 'een', 'betaling'],
    ['een', 'werk', 'dan', 'wel', 'het', 'verwezenlijken', ',', 'met', 'welke', 'middelen', 'dan', 'ook', ',', 'van', 'een', 'werk', 'dat', 'voldoet', 'aan', 'de', 'eisen', 'van', 'de', 'aanbestedende', 'dienst', 'of', 'het', 'speciale', '-', 'sectorbedrijf', 'die', 'een', 'beslissende', 'invloed', 'uitoefenen', 'op', 'het', 'soort', 'werk', 'of', 'op', 'het', 'ontwerp', 'van', 'het', 'werk', ','],
    ['een', 'instantie', 'die', 'conformiteitsbeoordelingsactiviteiten', 'verricht', 'en', 'die', 'geaccrediteerd', 'is', 'overeenkomstig', 'verordening', '(', 'EG', ')', 'nr', '.', '765', '/', '2008', 'van', 'het', 'Europees', 'Parlement', 'en', 'de', 'Raad', 'van', '9', 'juli', '2008', 'tot', 'vaststelling', 'van', 'de', 'eisen', 'inzake', 'accreditatie', 'en', 'markttoezicht', 'betreffende', 'het', 'verhandelen', 'van', 'producten', 'en', 'tot', 'intrekking', 'van', 'verordening', '(', 'EEG', ')', 'nr', '.', '339', '/', '93', '(', 'PbEU', '2008', ',', 'L', '218', ')'],
    ['de', 'gemeenschappelijke', 'woordenlijst', 'overheidsopdrachten', ',', 'vastgesteld', 'bij', 'verordening', '(', 'EG', ')', 'nr', '.', '2195', '/', '2002', 'van', 'het', 'Europees', 'Parlement', 'en', 'de', 'Raad', 'van', '5', 'november', '2002', 'betreffende', 'de', 'gemeenschappelijke', 'woordenlijst', 'overheidsopdrachten', '(', 'CPV', ')', '(', 'PbEG', '2002', ',', 'L', '340', ')'],
    ['een', 'ieder', 'die', 'diensten', 'op', 'de', 'markt', 'aanbiedt'],
    ['een', 'elektronisch', 'proces', 'voor', 'het', 'doen', 'van', 'gangbare', 'aankopen', 'van', 'werken', ',', 'leveringen', 'of', 'diensten', ',', 'waarvan', 'de', 'kenmerken', 'wegens', 'de', 'algemene', 'beschikbaarheid', 'op', 'de', 'markt', 'voldoen', 'aan', 'de', 'behoeften', 'van', 'de', 'aanbestedende', 'dienst', 'of', 'het', 'speciale', '-', 'sectorbedrijf', ',', 'dat', 'is', 'beperkt', 'in', 'de', 'tijd', 'en', 'gedurende', 'de', 'gehele', 'looptijd', 'openstaat', 'voor', 'ondernemers', 'die', 'aan', 'de', 'eisen', 'voor', 'toelating', 'tot', 'het', 'systeem', 'voldoen'],
    ['een', 'middel', 'waarbij', 'gebruik', 'wordt', 'gemaakt', 'van', 'elektronische', 'apparatuur', 'voor', 'gegevensverwerking', '(', 'met', 'inbegrip', 'van', 'digitale', 'compressie', ')', 'en', 'gegevensopslag', ',', 'alsmede', 'van', 'verspreiding', ',', 'overbrenging', 'en', 'ontvangst', 'door', 'middel', 'van', 'draden', ',', 'straalverbindingen', ',', 'optische', 'middelen', 'of', 'andere', 'elektromagnetische', 'middelen'],
    ['het', 'elektronische', 'systeem', 'voor', 'aanbestedingen', ',', 'bedoeld', 'in'],
    ['factuur', 'die', 'is', 'opgesteld', ',', 'verzonden', 'en', 'ontvangen', 'in', 'een', 'gestructureerde', 'elektronische', 'vorm', 'die', 'automatische', 'en', 'elektronische', 'verwerking', 'ervan', 'mogelijk', 'maakt'],
    ['een', 'zich', 'herhalend', 'elektronisch', 'proces', 'voor', 'de', 'presentatie', 'van', 'nieuwe', ',', 'verlaagde', 'prijzen', 'of', 'van', 'nieuwe', 'waarden', 'voor', 'bepaalde', 'elementen', 'van', 'de', 'inschrijvingen', ',', 'dat', 'plaatsvindt', 'na', 'de', 'eerste', 'volledige', 'beoordeling', 'van', 'de', 'inschrijvingen', 'en', 'dat', 'klassering', 'op', 'basis', 'van', 'automatische', 'beoordelingsmethoden', 'mogelijk', 'maakt'],
    ['een', 'activiteit', 'die', 'permanent', 'plaatsvindt', 'op', 'een', 'van', 'de', 'volgende', 'wijzen', ':', 'gecentraliseerde', 'aankoopactiviteit', ':'],
    ['de', 'verwerving', 'van', 'leveringen', 'of', 'diensten', 'die', 'bestemd', 'zijn', 'voor', 'aanbestedende', 'diensten', 'of', 'speciale', '-', 'sectorbedrijven'],
    ['het', 'plaatsen', 'van', 'overheidsopdrachten', 'respectievelijk', 'speciale', '-', 'sectoropdrachten', 'die', 'bestemd', 'zijn', 'voor', 'aanbestedende', 'diensten', 'of', 'speciale', '-', 'sectorbedrijven'],
    ['een', 'ondernemer', 'die', 'heeft', 'verzocht', 'om', 'een', 'uitnodiging', ',', 'of', 'is', 'uitgenodigd', ',', 'om', 'deel', 'te', 'nemen', 'aan', 'een', 'niet', '-', 'openbare', 'procedure', ',', 'een', 'procedure', 'van', 'de', 'concurrentiegerichte', 'dialoog', ',', 'een', 'mededingingsprocedure', 'met', 'onderhandeling', ',', 'een', 'procedure', 'van', 'het', 'innovatiepartnerschap', ',', 'een', 'onderhandelingsprocedure', 'met', 'aankondiging', ',', 'een', 'onderhandelingsprocedure', 'zonder', 'aankondiging', 'of', 'een', 'procedure', 'voor', 'de', 'gunning', 'van', 'een', 'concessieopdracht'],
    ['de', 'keuze', 'van', 'de', 'aanbestedende', 'dienst', 'of', 'het', 'speciale', 'sectorbedrijf', 'voor', 'de', 'ondernemer', 'met', 'wie', 'hij', 'voornemens', 'is', 'de', 'overeenkomst', 'waarop', 'de', 'procedure', 'betrekking', 'had', 'te', 'sluiten', ',', 'waaronder', 'mede', 'wordt', 'verstaan', 'de', 'keuze', 'om', 'geen', 'overeenkomst', 'te', 'sluiten']
]



# Apply the model to each token list and print the formatted output
for i, tokens in enumerate(token_lists):
    result_labels = classify_tokens(tokens)
    formatted_output = f"{i + 1}. Tokens: {tokens}\n   Labels: {result_labels}"
    print(formatted_output)
