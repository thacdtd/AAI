__author__ = 'thac'

import scipy.io as sio

def read_file():
    print "abc"
    with open("higgs-activity_time.txt", "r") as ins:
        array = []
        i = 0
        for line in ins:
            i += 1
            li1 = line.split()[0]
            li2 = line.split()[1]
            if li1 == '314' or li2 == '314':
                array.append(line);
        print i
    return array;

def read_file_sn():
    with open("higgs-social_network.edgelist", "r") as ins:
        i = 0
        for line in ins:
            i += 1
        print i

def load_nips():
    mat_contents = sio.loadmat('nips_1-17.mat')
    docs_authors = mat_contents["docs_authors"]
    print docs_authors

def load_kinship2():
    with open("kinship2.data", "r") as ins:
        array = []
        i = 0
        for line in ins:
            li1 = line.split(', ')[0]
            #li2 = line.split(', ')[1]
            if li1 != '\n' and li1 not in array:
                array.append(li1)
            if li1 != '\n':
                li2 = line.split(', ')[1]
                if li2 not in array:
                    array.append(li2)
            i += 1
    return array

a = load_kinship2()
print a
print len(a)