import numpy as np
from scipy.stats import norm

class NaiveBayesClassifier:
    def __init__(self, nb_features, cate_indices, cate_classes):
        self.cate_indices = cate_indices
        self.cate_classes = cate_classes # classes of categorical features
        self.nume_indices = list(set(np.arange(nb_features))-set(cate_indices))
    
    
    # for each label class, create a list of dictionaries to store 
    # parameters of categorical features
    def intialiseCateParameters(self):
        para = []
        for i in range(len(self.cate_indices)):
            p_temp = {}
            for j in range(len(self.cate_classes[i])):
                p_temp[self.cate_classes[i][j]] = 1
            para.append(p_temp)
        return para


    def fit(self, X, y):    
        N = y.size
        self.labels = np.unique(y)
        self.hash_table = []
        self.prior_para = []
        self.cate_para = []
        self.nume_para = []
        for i in range(self.labels.size):
            temp = np.nonzero(y==self.labels[i])[0]
            self.hash_table.append(temp)
            # for numerical features
            means = np.mean(X[temp, :][:,self.nume_indices], axis=0)
            
            stds = np.std(X[temp, :][:,self.nume_indices], axis=0, ddof=1)
            assert(np.all(stds!=0))
            self.nume_para.append([means,stds])
            # for categorical features
            temp_cate_para = self.intialiseCateParameters()
            for j in range(len(self.cate_indices)):
                for k in range(temp.size):
                    temp_cate_para[j][X[temp[k], self.cate_indices[j]]] += 1
                for key, value in temp_cate_para[j].items():
                    temp_cate_para[j][key] = np.log(temp_cate_para[j][key]/temp.size)
            self.cate_para.append(temp_cate_para)
            # for class prior probabilities
            self.prior_para.append(temp.size/N)


    def normal_log_probability(self, x, means, stds):
        stds = [np.clip(v, 0.00001, 10000000) for v in stds]
        temp = [np.log(norm.pdf(x[i], loc=means[i], scale=stds[i])) for i in range(x.size)]
        return np.sum(temp)
    
    
    def predict(self, X):
        class_log_probas = []
        for i in range(self.labels.size):
            temp_log_proba = np.zeros(X.shape[0])
            for j in range(X.shape[0]):
                # for numerical features
                temp_log_proba[j] += self.normal_log_probability(
                                        X[j, self.nume_indices],
                                        *self.nume_para[i])
                # for categorical features
                for k in range(len(self.cate_indices)):
                    temp_log_proba[j] += self.cate_para[i][k][X[j,
                                  self.cate_indices[k]]]
            temp_log_proba += self.prior_para[i]
            class_log_probas.append(temp_log_proba)
        class_log_probas = np.array(class_log_probas)
        predicted_classes = self.labels[np.argmax(class_log_probas, axis=0)]
        return predicted_classes
        
    
    # return mean accuracy
    def score(self, X, y):
        predicted_classes = self.predict(X)
        return np.sum(predicted_classes==y)/y.size





