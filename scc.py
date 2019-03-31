import csv, random
from itertools import zip_longest
from nltk import word_tokenize as tokenize
import nltk
#nltk.data.path.append("C:/Users/ks100/AppData/Roaming/nltk_data")
sent = "Mondays are boring."
print(tokenize(sent))

class question:
    
    def __init__(self,aline):
        self.fields=aline
    
    def get_field(self,field):
        return self.fields[question.colnames[field]]
    
    def add_answer(self,fields):
        self.answer=fields[1]
   
    def chooseA(self):
        return("a")

    def rand(self):
        return random.choice(["a","b","c","d","e"])

    def unigram(self, model):
        options = ["a", "b", "c", "d", "e"]
        opt_dict= dict(zip_longest(self.fields[2:], options))

        tokens=self.fields[2:]
        probs = dict()
        for token in tokens:
            probs[token]=model.get_prob(token,method="unigram")

        return opt_dict[random.choice([token for token, prob in probs.items() if prob==max(probs.values())])]

    def left_context(self, model):
        options=["a", "b", "c", "d", "e"]
        opt_dict=dict(zip_longest(self.fields[2:],options))

        tokens=self.fields[2:]
        probs=dict()
        for token in tokens:
            context=get_left_context(tokenize(self.fields[1]), 1)[0]
            bigram=model.get_prob(context, method="bigram")
            prob=bigram.get(token, bigram.get("_UNK", 0))
            probs[token]=prob
        return opt_dict[random.choice([token for token, prob in probs.items() if prob==max(probs.values())])]

    def right_context(self, model):
        options=["a", "b", "c", "d", "e"]
        opt_dict = dict(zip_longest(self.fields[2:],options))
        tokens=self.fields[2:]
        probs=dict()
        for token in tokens:
            context=get_right_context(tokenize(self.fields[1]), 1)[0]
            bigram=model.get_prob(token, method="bigram")
            prob=bigram.get(context, bigram.get("_UNK", 0))
            probs[token]=prob
        return opt_dict[random.choice([token for token, prob in probs.items() if prob==max(probs.values())])]


    def left_and_right_context(self, model):
        options=["a","b","c","d","e"]
        opt_dict=dict(zip_longest(self.fields[2:],options))
        tokens=self.fields[2:]
        probs=dict()
        for token in tokens:
            left_context=get_left_context(tokenize(self.fields[1]), 1)[0]
            right_context=get_right_context(tokenize(self.fields[1]), 1)[0]
            left_bigram=model.get_prob(left_context, method="bigram")
            left_prob=left_bigram.get(token, left_bigram.get("_UNK", 0))
            right_bigram=model.get_prob(token, method="bigram")
            right_prob=right_bigram.get(right_context, right_bigram.get("UNK", 0))
            prob=left_prob*right_prob
            probs[token]=prob
        return opt_dict[random.choice([token for token, prob in probs.items() if prob==max(probs.values())])]

    def trigram(self, model):
        options=["a","b","c","d","e"]
        opt_dict=dict(zip_longest(self.fields[2:],options))
        tokens=self.fields[2:]
        probs=dict()
        for token in tokens:
            context=get_left_context(tokenize(self.fields[1]), 2)
            trigram=model.get_prob(context[0], method="trigram")
            #inner=trigram.get(context[1], trigram.get("_UNK"))
            #inner2=inner.get(token, trigram.get(context[1]).get("_UNK"))
            prob=trigram.get(context[1], trigram.get("_UNK", model.trigram.get("_UNK", 0).get("_UNK", 0))).get(token, trigram.get(context[1], trigram.get("_UNK", model.trigram.get("_UNK", 0).get("_UNK", 0))).get("_UNK", model.trigram.get("_UNK", 0).get("_UNK", 0).get("_UNK", 0)))
            probs[token]=prob
        return opt_dict[random.choice([token for token, prob in probs.items() if prob==max(probs.values())])]
    
    def predict(self,model=None,method="chooseA"):
        #eventually there will be lots of methods to choose from
        if method=="chooseA":
            return self.chooseA()
        elif method=="random":
            return self.rand()
        elif method=="unigram":
            return self.unigram(model=model)
        elif method=="left":
            return self.left_context(model=model)
        elif method=="right":
            return self.right_context(model=model)
        elif method=="both":
            return self.left_and_right_context(model=model)
        elif method=="trigram":
            return self.trigram(model=model)
        else:
            print("Method not implemented.")
            return -1
        
    def predict_and_score(self,model=None,method="chooseA"):
        
        #compare prediction according to method with the correct answer
        #return 1 or 0 accordingly
        prediction=self.predict(model=model,method=method)
        if prediction ==self.answer:
            return 1
        else:
            return 0

class scc_reader:
    
    def __init__(self,qs,ans):
        self.qs=qs
        self.ans=ans
        self.read_files()
        
    def read_files(self):
        
        #read in the question file
        with open(self.qs) as instream:
            csvreader=csv.reader(instream)
            qlines=list(csvreader)
        
        #store the column names as a reverse index so they can be used to reference parts of the question
        question.colnames={item:i for i,item in enumerate(qlines[0])}
        
        #create a question instance for each line of the file (other than heading line)
        self.questions=[question(qline) for qline in qlines[1:]]
        
        #read in the answer file
        with open(self.ans) as instream:
            csvreader=csv.reader(instream)
            alines=list(csvreader)
            
        #add answers to questions so predictions can be checked    
        for q,aline in zip(self.questions,alines[1:]):
            q.add_answer(aline)
        
    def get_field(self,field):
        return [q.get_field(field) for q in self.questions] 
    
    def predict(self,method="chooseA"):
        return [q.predict(method=method) for q in self.questions]
    
    def predict_and_score(self,model=None, method="chooseA"):
        scores=[q.predict_and_score(model=model,method=method) for q in self.questions]
        return sum(scores)/len(scores)


def get_left_context(sent_tokens,window,target="_____"):
    found=-1
    for i,token in enumerate(sent_tokens):
        if token==target:
            found=i
            break
    if found>-1:
        return sent_tokens[i-window:i]
    else:
        return []

def get_right_context(sent_tokens, window, target="_____"):
    found=-1
    for i, token in enumerate(sent_tokens):
        if token==target:
            found=i
            break
    if found>-1:
        return sent_tokens[i+1:i+window+1]
    else:
        return []