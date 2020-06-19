import re
import csv
import random 
from tqdm import tqdm

csv.field_size_limit(500 * 1024 * 1024)

def get_stopwords():
    '''
    Get English stop words list
    '''
    with open('stopwords.txt','r') as f:
        lines = f.readlines()
        words = [line.strip() for line in lines]
        return words

def get_lemma_dict():
    '''
    Get the dict for lemmaziation
    '''
    with open('lemma.txt','r') as f:
        lines = f.readlines()
        dic = {}
        for line in lines:
            word1,word2 = line.split()
            word2 = word2.strip()
            dic[word2] = word1
    return dic 

lemma_dic = get_lemma_dict()

def clean_str(string):
    """
    Tokenization/string cleaning for raw text
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def read_csv(filename):
    '''
    Get text features and labels from a csv file
    '''
    with open(filename,'r',) as f:
        reader = csv.reader(f)
        items = []
        labels = []
        for row in reader:
            if row[0] == 'data':
                continue
            item,label = row[0],row[1]
            item = clean_str(item)
            label = int(label)
            items.append(item)
            labels.append(label)
            
    return items,labels



def build_vocab(filename):
    '''
    Build the vocabulary from a training csv file
    '''
    items,_ = read_csv(filename)
    stopwords = get_stopwords()
    dictionary = {}
    for item in tqdm(items):
        words = item.split()
        for word in words:
            if word in lemma_dic.keys():  # lenmmaziation here 
                word = lemma_dic[word]
            if len(word) >= 3 and word not in stopwords:
                dictionary[word] = dictionary.get(word,0) + 1 
    dictionary = sorted(dictionary.items(),key=lambda item:item[1],reverse=True)
    return dictionary





def get_features(train_file,test_file,demension=10000):
    '''
    extract features and labels from training and testing files
    demension(int): The number of the demension of the feature
    '''
    print("Building dictionary now……")
    dictionary = build_vocab(train_file)[:demension]
    print("Dictionary is done now")
    word2id = {}
    for id,item in enumerate(dictionary):
        word2id[item[0]] = id
    train_items,train_labels = read_csv(train_file)
    test_items,test_labels = read_csv(test_file)
    train_features = [ set() for i in range(len(train_items))]
    test_features = [ set() for i in range(len(test_items))]
    print("extracting training features")
    for id, item in tqdm(enumerate(train_items)):
        words = item.split()
        for word in words:
            if word in word2id.keys():
                train_features[id].add(word2id[word])
    print("extracting testing features")
    for id,item in tqdm(enumerate(test_items)):
        words = item.split()
        for word in words:
            if word in word2id.keys():
                test_features[id].add(word2id[word])
    return train_features,train_labels,test_features,test_labels,word2id

def get_shuffle(list1,list2):
    '''
    Get training samples and labels shuffled
    '''
    merge_list = [[list1[i],list2[i]] for i in range(len(list1))]
    random.shuffle(merge_list)
    list1 = [merge_list[0] for i in range(len(list1))]
    list2 = [merge_list[1] for i in range(len(list2))]
    return list1,list2



