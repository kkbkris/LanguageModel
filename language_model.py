import os, random, math

TRAINING_DIR = "res/lab3resources/sentence-completion/Holmes_Training_Data"  # this needs to be the parent directory for the training corpus


def get_training_testing(training_dir=TRAINING_DIR, split=0.5):
    filenames = os.listdir(training_dir)
    n = len(filenames)
    print("There are {} files in the training directory: {}".format(n, training_dir))
    # random.seed(53) #if you want the same random split every time
    random.shuffle(filenames)
    index = int(n * split)
    return (filenames[:index], filenames[index:])


#trainingfiles, heldoutfiles = get_training_testing()

from nltk import word_tokenize as tokenize
import operator


class language_model():

    def __init__(self, trainingdir=TRAINING_DIR, files=[], threshold=2, kn=True):
        self.training_dir = trainingdir
        self.files = files

        self.train(threshold, kn)

    def train(self, threshold, kn):
        self.unigram = {}
        self.bigram = {}
        self.trigram = {}
        self.quadrigram = {}
        self._processfiles()
        unigrams = self.unigram.copy()
        bigrams = self.bigram.copy()
        trigrams = self.trigram.copy()
        quadrigrams = self.quadrigram.copy()

        #deleting OOV words and tokens

        #unknowns for unigram
        for token, count in unigrams.items():
            if count < threshold:
                del self.unigram[token]
                self.unigram["_UNK"] = self.unigram.get("_UNK", 0) + 1
        self.bigram["_UNK"] = dict()

        #unknowns for bigram
        for token, innerdict in bigrams.items():
            inner_copy = innerdict.copy()
            for token1, count in inner_copy.items():
                if count < threshold:
                    del innerdict[token1]
                    innerdict["_UNK"] = innerdict.get("_UNK", 0) + 1
            inner_copy_new = self.bigram[token]
            if token not in self.unigram.keys():
                del self.bigram[token]
                for token1, count in inner_copy_new.items():
                    self.bigram["_UNK"][token1] = self.bigram["_UNK"].get(token1, 0) + count

        #unknowns for trigram
        self.trigram["_UNK"]=dict(dict())
        self.trigram["_UNK"]["_UNK"]=dict()
        for token, innerdict in trigrams.items():
            inner_copy = innerdict.copy()
            for token1, innerdict2 in inner_copy.items():
                inner_copy2 = innerdict2.copy()
                for token2, count in inner_copy2.items():
                    if count < threshold:
                        del innerdict2[token2]
                        innerdict2["_UNK"] = innerdict2.get("_UNK", 0) + 1
                inner_copy2_new = self.trigram[token][token1]
                if token1 not in self.unigram.keys():
                    del self.trigram[token][token1]
                    for token2, count in inner_copy2_new.items():
                        self.trigram[token]["_UNK"]=dict()
                        self.trigram[token]["_UNK"][token2] = self.trigram["_UNK"]["_UNK"].get(token2, 0) + count
            inner_copy_new = self.trigram[token]
            if token not in self.unigram.keys():
                del self.trigram[token]
                for token1, inner_dict in inner_copy_new.items():
                    for token2, count in inner_dict.items():
                        self.trigram["_UNK"][token1] = dict()
                        self.trigram["_UNK"][token1][token2] = self.trigram["_UNK"][token1].get(token2, 0) + count

        #unknowns for quadrigram
        self.quadrigram["_UNK"] = dict(dict(dict()))
        self.quadrigram["_UNK"]["_UNK"] = dict(dict())
        self.quadrigram["_UNK"]["_UNK"]["_UNK"] = dict()
        for token, innerdict in quadrigrams.items():
            inner_copy = innerdict.copy()
            for token1, innerdict2 in inner_copy.items():
                inner_copy2 = innerdict2.copy()
                for token2, innerdict3 in inner_copy2.items():
                    inner_copy3 = innerdict3.copy()
                    for token3, count in inner_copy3.items():
                        if count < threshold:
                            del innerdict3[token3]
                            innerdict3["_UNK"] = innerdict3.get("_UNK", 0) + 1
                    inner_copy3_new = self.quadrigram[token][token1][token2]
                    if token2 not in self.unigram.keys():
                        del self.quadrigram[token][token1][token2]
                        for token3, count in inner_copy3_new.items():
                            self.quadrigram[token][token1]["_UNK"] = dict()
                            self.quadrigram[token][token1]["_UNK"][token3] = self.quadrigram["_UNK"]["_UNK"]["_UNK"].get(token3, 0) + count
                inner_copy2_new = inner_copy2.copy()
                if token1 not in self.unigram.keys():
                    del self.quadrigram[token][token1]
                    for token2, inner_dict in inner_copy2_new.items():
                        for token3, count in inner_dict.items():
                            self.quadrigram[token]["_UNK"] = dict(dict())
                            self.quadrigram[token]["_UNK"][token2] = dict()
                            self.quadrigram["_UNK"]["_UNK"][token2] = dict()
                            self.quadrigram[token]["_UNK"][token2][token3] = self.quadrigram["_UNK"]["_UNK"][token2].get(token3, 0) + count
            inner_copy_new = inner_copy.copy()
            if token not in self.unigram.keys():
                del self.quadrigram[token]
                for token1, inner_dict in inner_copy_new.items():
                    for token2, inner_dict2 in inner_dict.items():
                        for token3, count in inner_dict2.items():
                            self.quadrigram["_UNK"][token1][token2] = dict()
                            self.quadrigram["_UNK"][token1][token2][token3] = self.quadrigram["_UNK"][token1][token2].get(token3, 0) + count



        print()
        print("-----------Checking for Zero Sums-----------")
        print()
        flag = False
        for token, inner_dict in self.trigram.items():

            if inner_dict.items()==[]:
                print("FOUND EMPTY INNER DICTIONARY")
                print(token)
                print(self.trigram[token])

            for token1, inner_dict2 in inner_dict.items():
                if len(inner_dict2)==0:
                    print("FOUND ZERO LENGTH DICTIONARY")
                    flag=True
                    print(token)
                    print(token1)
                    print(self.trigram[token])
                    print(self.trigram[token][token1])
                if sum(inner_dict2.values())==0:
                    print("FOUND INNER ZERO SUM")
                    flag=True
                    print(token)
                    print(token1)
                    print(self.trigram[token])
                    print(self.trigram[token][token1])

            if sum([sum(inner_dict2.values()) for token2, inner_dict2 in inner_dict.items()])==0:
                print("FOUND OUTER ZERO SUM")
                flag=True
                print(token)
                print(self.trigram[token])

        if not flag:
            print("Found no zero sums.")

        print("Checking for embedded values.")
        for token, inner_dict in self.quadrigram.items():
            for token1, inner_dict2 in inner_dict.items():
                for token2, inner_dict3 in inner_dict2.items():
                    continue
            print(sum([sum([len(inner_dict3) for token3, inner_dict3 in inner_dict2.items()]) for token2, inner_dict2 in inner_dict.items()]))


        print("Finished checking for embedded values.")
        #print(self.quadrigram)

        self.discount()

        #print(self.trigram["Thou"])
        for token, inner_dict in self.trigram.items():
            for token1, inner_dict2 in inner_dict.items():
                if "_DISCOUNT" not in self.trigram[token][token1].keys():
                    print("DISCOUNT keys")
                    print(token)
                    print(token1)
                    print(self.trigram[token])
                    print(self.trigram[token][token1])
                print(self.trigram[token][token1]["_DISCOUNT"])
                if sum(inner_dict2.values())==0:
                    print("Zero Sums")
                    print(token)
                    print(token1)
                    print(self.trigram[token])
                    print(self.trigram[token][token1])


        trigram_test = self.trigram
        trigram_test = {l:{k:{t:self.trigram[l][k]["_DISCOUNT"] for (t, v) in inner_dict.items()} for (k,inner_dict) in inner_dict2.items()} for (l, inner_dict2) in self.trigram.items()}
        print(trigram_test)
        print("TEST_SUCCESS")

        self._convert_to_probs()
        if kn:
            self.kneser_ney()

    def _processline(self, line):
        tokens = ["__START"] + tokenize(line) + ["__END"]
        prev_token = tokens[0]
        prev_tok2 = tokens[0]
        prev_tok3 = tokens[0]
        for token in tokens:
            #construct the quadrigram
            if prev_tok3 in self.quadrigram.keys():
                if prev_tok2 in self.quadrigram[prev_tok3].keys():
                    if prev_token in self.quadrigram[prev_tok3][prev_tok2].keys():
                        self.quadrigram[prev_tok3][prev_tok2][prev_token][token] = self.quadrigram[prev_tok3][prev_tok2][prev_token].get(token, 0) + 1
                    else:
                        self.quadrigram[prev_tok3][prev_tok2][prev_token] = {token: 1}
                else:
                    self.quadrigram[prev_tok3][prev_tok2] = {prev_token: {token: 1}}
            else:
                self.quadrigram[prev_tok3] = {prev_tok2: {prev_token: {token: 1}}}

            #construct the trigram
            if prev_tok2 in self.trigram.keys():
                if prev_token in self.trigram[prev_tok2].keys():
                    self.trigram[prev_tok2][prev_token][token] = self.trigram[prev_tok2][prev_token].get(token, 0) + 1
                else:
                    self.trigram[prev_tok2][prev_token] = {token: 1}
            else:
                self.trigram[prev_tok2] = {prev_token: {token: 1}}

            #construct the bigram
            if prev_token in self.bigram:
                self.bigram[prev_token][token] = self.bigram[prev_token].get(token, 0) + 1
            else:
                self.bigram[prev_token] = {token: 1}

            #construct the unigram
            self.unigram[token] = self.unigram.get(token, 0) + 1
            prev_tok2 = prev_token
            prev_token = token

    #def cons_bigram(self):


    def _processfiles(self):
        for afile in self.files:
            print("Processing {}".format(afile))
            try:
                with open(os.path.join(self.training_dir, afile)) as instream:
                    for line in instream:
                        line = line.rstrip()
                        if len(line) > 0:
                            self._processline(line)
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing {}: ignoring file".format(afile))

    def _convert_to_probs(self, discount="abs_disc"):

        self.unigram = {k: v / sum(self.unigram.values()) for (k, v) in self.unigram.items()}
        self.bigram = {
        k: {t: v / sum(innerdict.values()) + ((self.bigram[k]["_DISCOUNT"] / sum(innerdict.values())) * self.unigram[k])
            for (t, v) in innerdict.items()} for (k, innerdict) in self.bigram.items()}
        self.trigram = {l: {k: {t: math.exp((math.log((v/sum(inner_dict2.values())))
                                   +math.log((sum(inner_dict2.values())/
                                     sum([sum(inner_dict2.values()) for (k, inner_dict2) in inner_dict.items()])))
                                    +math.log(self.unigram[t])))
                       + ((self.trigram[l][k]["_DISCOUNT"]/sum([sum(inner_dict2.values()) for token2, inner_dict2 in inner_dict.items()]))*self.unigram[t])
                                for (t, v) in inner_dict2.items()}
                            for (k, inner_dict2) in inner_dict.items()}
                        for (l, inner_dict) in self.trigram.items()}
        self.quadrigram = {m: {l: {k: {t: math.exp(math.log((v/sum(inner_dict3.values())))+
                                           math.log((sum(inner_dict3.values())/sum([sum(inner_dict3.values()) for (k, inner_dict3) in inner_dict2.items()])))+
                                           math.log((sum([sum(inner_dict3.values()) for (k, inner_dict3) in inner_dict2.items()])/sum([sum([sum(inner_dict3.values()) for (k, inner_dict3) in inner_dict2.items()]) for (l, inner_dict2) in inner_dict.items()])))+
                                           math.log(self.unigram[m]))+
                                          ((self.quadrigram[m][l][k]["_DISCOUNT"]/sum([sum(inner_dict3.values()) for (token3, inner_dict3) in inner_dict2.items()]))*self.unigram[t])
                                       for (t, v) in inner_dict3.items()}
                                   for (k, inner_dict3) in inner_dict2.items()}
                               for (l, inner_dict2) in inner_dict.items()}
                           for (m, inner_dict) in self.quadrigram.items()}

        #(self.bigram[l][k]*(self.bigram[k][t]/sum([]))

                                    #v/sum(innerdict.values())
                                   #+ ((self.trigram[l][k]["_DISCOUNT"]/sum(innerdict.values()))*self.trigram[l][k])}}}

    def get_prob(self, token1, method="unigram"):
        if method == "unigram":
            return self.unigram.get(token1, self.unigram["_UNK"])
        elif method == "bigram":
            return self.bigram.get(token1, self.bigram["_UNK"])
        elif method == "trigram":
            return self.trigram.get(token1, self.trigram["_UNK"])
        elif method == "quadrigram":
            return self.quadrigram.get(token1, self.quadrigram["_UNK"])
        else:
            print("Not implemented: {}".format(method))
            return 0

    def most_probable(self, k=10, limit=10, terminator="."):
        sorted_unigram = list(sorted(self.unigram.items(), key=lambda kv: kv[1]))
        tokens = []
        if terminator:
            while True:
                token, prob = random.choice(sorted_unigram[:k])
                if token == terminator:
                    return tokens
                else:
                    tokens.append(token)
        else:
            for i in range(limit):
                token, prob = random.choice(sorted_unigram[:k])
                tokens.append(token)
            return tokens

    #discounting method
    def discount(self):
        #discounting on unigrams - useful to have
        self.unigram["_DISCOUNT"] = len(self.unigram) * 0.75

        #discounting on bigrams
        for token, innerdict in self.bigram.items():
            for token1, count in innerdict.items():
                self.bigram[token][token1] = self.bigram[token].get(token1, 0) - 0.75
            self.bigram[token]["_DISCOUNT"] = len(innerdict) * 0.75

        #discounting on trigrams
        for token, inner_dict in self.trigram.items():
            for token1, inner_dict2 in inner_dict.items():
                for token2, count in inner_dict2.items():
                    self.trigram[token][token1][token2] = self.trigram[token].get(token1, {token2: 0}).get(token2, 0) - 0.75
                self.trigram[token][token1]["_DISCOUNT"] = 0
                self.trigram[token][token1]["_DISCOUNT"] = len(inner_dict2) * 0.75
            self.trigram[token]["_DISCOUNT"] = dict()
            self.trigram[token]["_DISCOUNT"]["_DISCOUNT"] = sum([len(inner_dict2) for token2, inner_dict2 in inner_dict.items()]) * 0.75

        #discounting on quadrigrams
        for token, inner_dict in self.quadrigram.items():
            for token1, inner_dict2 in inner_dict.items():
                for token2, inner_dict3 in inner_dict2.items():
                    for token3, count in inner_dict3.items():
                        self.quadrigram[token][token1][token2][token3] = self.quadrigram[token].get(token1, {token2: {token3:0}}).get(token2, {token3: 0}).get(token3, 0) - 0.75
                    self.quadrigram[token][token1][token2]["_DISCOUNT"] = 0
                    self.quadrigram[token][token1][token2]["_DISCOUNT"] = len(inner_dict3) * 0.75
                self.quadrigram[token][token1]["_DISCOUNT"] = dict()
                self.quadrigram[token][token1]["_DISCOUNT"]["_DISCOUNT"] = sum([len(inner_dict3) for token3, inner_dict3 in inner_dict2.items()]) * 0.75
            print(sum([sum([len(inner_dict3) for token3, inner_dict3 in inner_dict2.items() if not token3=="_DISCOUNT"]) for token2, inner_dict2 in inner_dict.items()]))
            self.quadrigram[token]["_DISCOUNT"] = dict(dict())
            self.quadrigram[token]["_DISCOUNT"]["_DISCOUNT"] = dict()
            self.quadrigram[token]["_DISCOUNT"]["_DISCOUNT"]["_DISCOUNT"] = sum([sum([len(inner_dict3) for token3, inner_dict3 in inner_dict2.items() if not token3=="_DISCOUNT"]) for token2, inner_dict2 in inner_dict.items()]) * 0.75

    #kneser-ney smoothing method
    def kneser_ney(self):
        for token, innerdict in self.bigram.items():
            kn_prob = len([k for (k, v) in innerdict.items() if v > 0]) / sum(
                [len([k for (k, v) in d2.items() if v > 0]) for t, d2 in self.bigram.items()])
            # kn2_prob=len([k for (k,d) in self.bigram.items() if k==token and d[k] > 0])/sum([len([k for (k,v) in d2.items() if v > 0]) for t,d2 in self.bigram.items()])
            # print(kn_prob==kn2_prob)
            for t2, prob in innerdict.items():
                self.bigram[token][t2] = prob + (self.bigram[token]["_DISCOUNT"] * kn_prob)

    #function to calculate the total log probability
    def lp(self, corpus, model, method="unigram"):
        lm = model
        tot = 0
        if method == "unigram":
            for token, prob in lm.unigram.items():
                tot += math.log(prob)
            return tot
        elif method == "bigram":
            prev_tok = list(lm.bigram.keys())[0]
            for token in corpus:
                if token == '__START':
                    prev_tok = token
                    continue
                elif prev_tok == '__START':
                    if token not in lm.unigram.keys():
                        prob = lm.unigram["_UNK"]
                    else:
                        prob = lm.unigram[token]
                    tot += math.log(prob)
                elif token == "__END":
                    break
                else:
                    if prev_tok not in lm.bigram.keys():
                        if token not in lm.bigram["_UNK"].keys():
                            prob = lm.bigram["_UNK"].get("_UNK", 0)
                        else:
                            prob = lm.bigram["_UNK"].get(token, 0)
                    else:
                        if prev_tok in lm.bigram[prev_tok].keys():
                            prob = lm.bigram[prev_tok].get(token, 0)
                    tot += math.log(prob)
                prev_tok = token
            return tot

    def perplexity(self, corpus, model, method="unigram"):
        N = len(corpus)
        exponent = self.lp(corpus, model, method=method) / N
        if exponent < 0:
            return 1 - 1 / (1 + math.exp(exponent))
        else:
            return 1 / (1 + math.exp(-exponent))
