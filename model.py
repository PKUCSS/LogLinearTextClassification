import math
from tqdm import tqdm

class LogLinearModel:
    '''
    The Log-Linear Model for text categorization
    '''
    def __init__(self,lr=0.1,n_classes=20,n_features=10000,lemma =1e-4):
        '''
        lr:learning rate
        n_classes:number of classes
        n_features:the demension of features
        lemma:regularization coefficient
        '''
        self.w = [ [0.00001 for i in range(n_features)] for i in range(n_classes)]
        self.n_classes = n_classes
        self.n_features = n_features
        self.lemma = lemma
        self.lr = lr

    def predict(self,samples):
        '''
        Get predict result of given samples 
        '''
        labels = []
        scores_list = []
        for sample in samples:
                scores = []
                max_tmp = 0 
                for clazz in range(self.n_classes):
                    tmp_score = sum([id*self.w[clazz][id] for id in sample])
                    max_tmp = max(max_tmp,tmp_score)
                    scores.append(tmp_score)
                scores = [ score - max_tmp for score in scores]
                scores  = [math.exp(score) for score in scores]
                s = sum(scores)
                scores = [score/s for score in scores]
                label = 0
                max_score = scores[0]
                for i in range(1,self.n_classes):
                    if scores[i] > max_score:
                        max_score = scores[i]
                        label = i
                labels.append(label)
                scores_list.append(scores)
        return labels, scores_list

    def get_update(self,samples,labels):
        '''
        Get coefficients updated for a mini-batch of samples
        '''
        _,scores_list = self.predict(samples)
        gradient = [ [0.0 for _ in range(self.n_features)] for _ in range(self.n_classes)]
        for id,sample in enumerate(samples):
            for word_id in sample:
                for clazz in range(self.n_classes):
                    if clazz == labels[id]:
                        gradient[clazz][word_id] += 1.0 - 1.0*scores_list[id][clazz]
                    else:
                        gradient[clazz][word_id] += 0.0 - 1.0*scores_list[id][clazz]
        for i in range(self.n_classes):
            for j in range(self.n_features):
                self.w[i][j] += self.lr*gradient[i][j] - self.lemma*self.w[i][j]
        return 

    def train(self,samples,labels,batch_size=100):
        '''
        training for given data and the batch size
        '''
        idx = 0 
        for _ in tqdm(range(len(samples) // batch_size )):
            samples_batch = samples[idx:idx+batch_size]
            labels_batch = labels[idx:idx+batch_size]
            idx += batch_size 
            self.get_update(samples_batch,labels_batch)
            if idx >= len(samples):
                break
        return
    
    def test(self,samples,labels):
        '''
        test on given samples and labels
        '''
        pred,_ = self.predict(samples)
        correct_num = 0 
        for i in range(len(samples)):
            if pred[i] == labels[i]:
                correct_num = correct_num + 1

        return correct_num/len(samples)