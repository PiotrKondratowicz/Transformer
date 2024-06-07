import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def change_labels(ner_output, texts):

    changed_labels = []

    for text, ner_entities in zip(texts, ner_output):
        labels = ['O'] * len(text.split())
        for entity in ner_entities:
            entity_label = entity['entity_group']
            entity_start_index = entity['start']
            entity_end_index = entity['end']
            entity_words = text[entity_start_index:entity_end_index].split()

            word_start_index = len(text[:entity_start_index].split())
            word_end_index = word_start_index + len(entity_words)

            if word_start_index < len(labels) and word_end_index <= len(labels):
                labels[word_start_index] = 'B-' + str(entity_label)
                for i in range(word_start_index + 1, word_end_index):
                    labels[i] = 'I-' + str(entity_label)

        changed_labels.append(labels)

    return changed_labels


model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

validation_data = pd.read_csv('dev-0/in.tsv', sep='\t', names=["texts"], header=None)
validation_data_labels = pd.read_csv('dev-0/expected.tsv', sep='\t', names=["labels"], header=None)
test_data = pd.read_csv('test-A/in.tsv', sep='\t', names=["texts"], header=None)

texts_validation = validation_data["texts"].tolist()
texts_test = test_data["texts"].tolist()

ner_validation = ner_pipeline(texts_validation)
ner_test = ner_pipeline(texts_test)

predicted_labels_validation = change_labels(ner_validation, texts_validation)
predicted_labels_test = change_labels(ner_test, texts_test)

final_output_validation = [' '.join(labels) for labels in predicted_labels_validation]
final_output_test = [' '.join(labels) for labels in predicted_labels_test]

with open('dev-0/out.tsv', 'w') as f:
    for line in final_output_validation:
        f.write(line + '\n')

with open('test-A/out.tsv', 'w') as f:
    for line in final_output_test:
        f.write(line + '\n')

