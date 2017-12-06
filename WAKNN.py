import csv
import re
from stemmer import PorterStemmer
from collections import defaultdict
import operator
from numpy import *
import threading
import queue

class WAKNN(object):

    def __init__(self, word_size=100, k=10):
        #
        self.words = set()
        self.weight = []
        self.origin_documents = []
        self.documents = []
        self.labels = []
        self.word_size = word_size
        self.k = k
        self.factors = [0.2, 0.8, 1.5, 2.0, 4.0]

    # load text data for csv file
    def load(self, filename):
        with open(filename) as csvfile:
            data = csv.reader(csvfile)
            setData = list(data)

            # data preprocessing
            for index in range(1, len(setData)):
                label = str(setData[index][2])[:-1].split(',')
                # we only use document with one label
                if len(label) == 1 and label[0] is not '':
                    # save label of each document to labels array
                    self.labels.append(label[0])
                    # replace special chars with space
                    self.origin_documents.append(re.sub(r'[^\w]'," ", setData[index][5]).lower())

                if len(self.labels) > 500:
                    return

    # build related matrix
    def buildMatrix(self):
        # use suffix-stripping algorithm to stem word
        porter_strmmer = PorterStemmer()
        for index in range(0,len(self.origin_documents)):
            document = self.origin_documents[index]
            # change document in origin_document array to array of stemmed word
            self.origin_documents[index] = [porter_strmmer.stem(x, 0, len(x) - 1) for x in document.split()]

        # use 2000 most frequent words to generate words array
        temp_word = defaultdict(int)
        for document in self.origin_documents:
            for word in document:
                temp_word[word] += 1

        sorted_dict = sorted(temp_word.items(), key=operator.itemgetter(1))
        sorted_dict.reverse()
        self.words =  [x[0] for x in sorted_dict[0:self.word_size]]


        # build document array
        for index in range(0, len(self.origin_documents)):
            document = self.origin_documents[index]
            self.documents.append([])
            self.documents[index] = [document.count(word) for word in self.words]

        # print(self.documents[0], sum(self.documents[0]))

        # remove zero sum rows
        zeros = [i for i, value in enumerate(self.documents) if sum(value) == 0]
        for value in zeros[::-1]:
            del self.labels[value]
            del self.documents[value]

        # zeros = [i for i, value in enumerate(self.documents) if sum(value) == 0]

        print(len(self.origin_documents), len(self.words), len(self.documents), self.words)

    # Normalize word frequencies in each document such that they add up to 1.0.
    def normalize(self):
        for index in range(0, len(self.documents)):
            row_sum = sum(self.documents[index])
            if row_sum > 0:
                self.documents[index] = [x / row_sum for x in self.documents[index]]

        # print(sum(self.documents[0]), self.documents[0])

    # Initialize weight vector W.
    def initializeWeight(self):
        self.weight = [1.0 for i in range(0, len(self.words))]
        print(self.weight)


    # cos(X, Y, W)
    def weightedCosine(self, X, Y, W):
        mx = reshape(X, (1, len(X)))
        my = reshape(Y, (1, len(Y)))
        mw = reshape(W, (1, len(W)))

        molecular = sum(mx * mw * my * mw)
        denominator = sqrt(sum(mx * mw * mx * mw)) * sqrt(sum(my * mw * my * mw))

        return molecular / denominator

    # return k nearest neighbors
    def knn(self, d, W):
        similarity_array = list()

        for document in self.documents:
            similarity_array.append(self.weightedCosine(d, document,W))


        # print(similarity_array)
        similarity_array = array(similarity_array)

        neighbors = similarity_array.argsort()[-self.k:][::-1]
        # print(self.labels)
        return [self.labels[x] for x in neighbors]

    # def classify(self):


    # simple object function
    # add up the number of training documents that are correctly classified using their k-nearest neighbors
    def simpleObj(self, W):
        result = 0

        for index, document in enumerate(self.documents):
            neighbors = self.knn(document, W)

            label = self.labels[index]

            count_array = [neighbors.count(x) for x in neighbors]
            max_index, max_value = max(enumerate(count_array), key=operator.itemgetter(1))

            if label == neighbors[max_index]:
                result += 1
            else:
                print(index, label, neighbors[max_index], neighbors)
        return result

    # Objective function with majority percentage
    # sum of similarities of d’s true neighbors is at least p percentage of the total similarity sum
    def majorityObj(self, p, W):
        result = 0
        for index, document in enumerate(self.documents):
            neighbors = self.knn(document, W)

            label = self.labels[index]

            count_array = [neighbors.count(x) for x in neighbors]
            max_index, max_value = max(enumerate(count_array), key=operator.itemgetter(1))

            if label == neighbors[max_index]:
                if max_value / self.k > p:
                    result += 1
            else:
                # print(index, label, neighbors[max_index], neighbors)
                pass

        return result

    # update an item of weight array with a new value
    def updateWeight(self, i, newValue):
        self.weight[i] = newValue

    # return a new weight array by changing one item of old weight array without changing original array
    def newWeight(self, i, factor):
        new_weight = list(self.weight)

        new_weight[i] = new_weight[i] * factor

        return new_weight


    # training weight array in single thread
    def trainingWeight(self, p):
        factors = [0.2, 0.8, 1.5, 2.0, 4.0]

        new_objs = []

        # get all new Majobj value by multiply a factor to origin weight item for each item
        for index in range(0, len(self.weight)):
            new_objs.append([self.majorityObj(p, self.newWeight(index, x)) for x in factors])

        return new_objs


    # training weight array in multi-thread
    def trainingWeightMulti(self, p):
        n = self.n
        result = queue.Queue()
        threads = []
        for i in range(1, n + 1):
            t = threading.Thread(target=self.majorityObjMulti, args=(p, i, n, result), name=str(i))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()
        result = list(result.queue)

        print(result)

        final_result = []

        for i in range(1, n + 1):
            for r in result:
                if r[0] == i:
                    final_result.extend(r[1])

        print(final_result)

        return final_result

    # compute sub majobj value , this function will be target of threads
    def majorityObjMulti(self, p, index, n, result):
        factors = [0.2, 0.8, 1.5, 2.0, 4.0]
        sub_result = []

        step = int(self.word_size / n)

        for i in range((index - 1) * step , step * index):

            if i >= self.word_size:
                break
            else:
                sub_result.append([self.majorityObj(p, self.newWeight(i, x)) for x in factors])



        result.put((index, sub_result))

        return result

    # training algorithm with p specified, you can choose single thread version or multi-thread version
    def training(self,p,version = "multi", threads = 4):

        if version == "multi":
            f = self.trainingWeightMulti
            self.n = threads
        else:
            f = self.trainingWeight

        max_obj = self.majorityObj(p, self.weight)

        update = True
        round = 0

        while update:
            round += 1
            update = False
            result = f(p)
            print("round-{} start".format(round))
            for index in range(0,len(result)):
                max_index, max_value = max(enumerate(result[index]), key=operator.itemgetter(1))

                if max_value > max_obj:
                    self.updateWeight(index, self.factors[max_index] * self.weight[index])
                    update = True

                max_obj = self.majorityObj(p, self.weight)

            print("round-{} finish".format(round))

        print("final_result", self.weight)
        return "done"










