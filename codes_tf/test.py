import pickle

ids_to_labels = {
    0: "B-LOCATION",
    1: "B-MISCELLANEOUS",
    2: "B-ORGANIZATION",
    3: "B-PERSON",
    4: "I-LOCATION",
    5: "I-MISCELLANEOUS",
    6: "I-ORGANIZATION",
    7: "I-PERSON",
    8: "O"
}

reversed_dict = {v: k for k, v in ids_to_labels.items()}


with open('ids_to_labels.pkl', 'wb') as file:
    pickle.dump(ids_to_labels, file)


with open('labels_to_ids.pkl', 'wb') as file:
    pickle.dump(reversed_dict, file)
