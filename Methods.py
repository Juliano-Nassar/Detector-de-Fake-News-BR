import random
from copy import copy
import pandas as pd

def aug_remove_words(text,mapping = {},remove_pct = 0.3):
    
    text_splitted = text.split(' ')
    total_remove = int(len(text_splitted)*remove_pct)

    for i in range(total_remove):
        remove_index = random.randint(0,len(text_splitted)-1)
        del text_splitted[remove_index]
        
    return ' '.join(text_splitted)

def aug_swap_words_pos(text,mapping = {},swap_pct = 0.4):

    text_splitted = text.split(' ')

    max_switchs = int(len(text_splitted)/2)
    n_switchs =int(max_switchs*swap_pct)

    indexes = list(range(len(text_splitted)))
   
    for i in range(n_switchs):
        aux_indexes = indexes.copy()
            
        split_index_1 = random.randint(0,len(aux_indexes)-1)
        index_1 = aux_indexes[split_index_1]
        a = copy(text_splitted[split_index_1])
        del aux_indexes[split_index_1]

        split_index_2 = random.randint(0,len(aux_indexes)-1)
        index_2 = aux_indexes[split_index_2]
        b = copy(text_splitted[index_2])
        del aux_indexes[split_index_2]

        text_splitted[split_index_1] = b
        text_splitted[split_index_2] = a
    
    return ' '.join(text_splitted)

def aug_replace_by_synonym(text,mapping,replace_pct=0.6):
    text_splitted = text.split(' ')

    aux = 0
    keys = mapping.keys()

    # Get only words indexes inside synonyms base
    changeable_words = []
    for word in text_splitted:
        if word in keys:
            changeable_words.append(aux)
        aux+=1

    n = len(changeable_words)
    n_max = n*replace_pct

    words_replaced = 0
    # Change randomly the word by one synonym
    if len(changeable_words)>0:
        resampled_words = random.sample(changeable_words,len(changeable_words))
        for word_index in resampled_words:
            synonyms = mapping[text_splitted[word_index]]
            text_splitted[word_index] = random.choice(synonyms)
            words_replaced+=1
            
            if words_replaced >= n_max:
                break
                
    return ' '.join(text_splitted)

def aug_insert_by_synonym(text,mapping,replace_pct=0.3):
    text_splitted = text.split(' ')

    aux = 0
    keys = mapping.keys()

    # Get only words indexes inside synonyms base
    changeable_words = []
    for word in text_splitted:
        if word in keys:
            changeable_words.append(word)

    n = len(text_splitted)
    n_max = int(n*replace_pct)

    words_replaced = 0
    # Insert randomly a word by one synonym
    if len(changeable_words)>0:
        single_words = list(set(changeable_words))
        
        for i in range(n_max):
            word = random.choice(single_words)
            synonyms = mapping[word]
            pos = random.randint(0,len(text_splitted))
            
            text_splitted.insert(pos,random.choice(synonyms))
                
    return ' '.join(text_splitted)

def apply_aug(text,mapping):
    functions = [aug_remove_words,aug_swap_words_pos,aug_replace_by_synonym,aug_insert_by_synonym]
    func_chosen = random.choice(functions)
    
    text_augmented = func_chosen(text,mapping = mapping)
    return text_augmented

def create_agumented_data(train_dataset,mapping,RANDOM_SEED=42):
    random.seed(RANDOM_SEED)
    augmented_dataset = [train_dataset]
    for i in range(9):
        aux_dataset = train_dataset.copy()
        aux_dataset['preprocessed_news'] = aux_dataset['preprocessed_news'].apply(apply_aug,args=[mapping])

        augmented_dataset.append(aux_dataset.copy())

    final_df = pd.concat(augmented_dataset).reset_index(drop=True)
    return final_df