import sys
import csv


class NaiveBayes(object):

    def __init__(self, trainingfile=sys.argv[1], testfile=sys.argv[2], label1='+1', label2='-1'):
        self.trainingFile = trainingfile
        self.testFile = testfile
        self.label1 = label1
        self.label2 = label2
        index, val = self._parse_highest()
        self.countl1 = 0
        self.countl2 = 0
        self.model = dict()
        self.model[self.label1] = {}
        self.model[self.label2] = {}
        for x in range(1, index+1):
            self.model[self.label1][x] = {}
            self.model[self.label2][x] = {}
            for y in range(1, val+1):
                self.model[self.label1][x][y] = 0
                self.model[self.label2][x][y] = 0

    def train(self):
        with open(self.trainingFile, 'r') as data:
            data = csv.reader(data, delimiter=' ')
            for row in data:
                if row[0] == self.label1:
                    self.countl1 += 1
                    for pair in row[1:]:
                        attr = self._split(pair)
                        self.model[row[0]][attr[0]][attr[1]] += 1
                if row[0] == self.label2:
                    self.countl2 += 1
                    for pair in row[1:]:
                        attr = self._split(pair)
                        self.model[row[0]][attr[0]][attr[1]] += 1

    def test(self):
        self._test(self.trainingFile)
        self._test(self.testFile)

    def _test(self, file):
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        with open(file, 'r') as data:
            data = csv.reader(data, delimiter=' ')
            for row in data:
                pairs = []
                actual = row[0]
                prediction = ''
                for pair in row[1:]:
                    attr = self._split(pair)
                    pairs.append(attr)
                pvals = self._calculate(pairs)
                if pvals[0] >= pvals[1]:
                    prediction = self.label1
                else:
                    prediction = self.label2
                if actual == self.label1 and prediction == self.label1:
                    tp += 1
                if actual == self.label1 and prediction == self.label2:
                    fn += 1
                if actual == self.label2 and prediction == self.label1:
                    fp += 1
                if actual == self.label2 and prediction == self.label2:
                    tn += 1
            print('{} {} {} {}'.format(tp, fn, fp, tn))

    def _calculate(self, pairs):
        p1 = 1
        p2 = 1
        for attr in pairs:
            p1 *= self.model[self.label1][attr[0]][attr[1]] / self.countl1
            p2 *= self.model[self.label2][attr[0]][attr[1]] / self.countl2
        total = self.countl1 + self.countl2
        p1 *= (self.countl1/total)
        p2 *= (self.countl2 / total)
        return p1, p2

    def __str__(self):
        return "Class1\n{}\nClass2\n{}\n".format(str(self.model[self.label1]), str(self.model[self.label2]))

    def _parse_highest(self):
        highestVal = 0
        highestIndex = 0
        with open(self.trainingFile, 'r') as data:
            data = csv.reader(data, delimiter=' ')
            for row in data:
                for pair in row[1:]:
                    attr = NaiveBayes._split(pair)
                    if attr[0] > highestIndex:
                        highestIndex = attr[0]
                    if attr[1] > highestVal:
                        highestVal = attr[1]
        with open(self.testFile, 'r') as data:
            data = csv.reader(data, delimiter=' ')
            for row in data:
                for pair in row[1:]:
                    attr = NaiveBayes._split(pair)
                    if attr[0] > highestIndex:
                        highestIndex = attr[0]
                    if attr[1] > highestVal:
                        highestVal = attr[1]
        return highestIndex, highestVal

    @staticmethod
    def _split(attr):
        pair = attr.split(':')
        pair[0] = int(pair[0])
        pair[1] = int(pair[1])
        return pair


if __name__ == "__main__":
    model = NaiveBayes()
    model.train()
    model.test()


