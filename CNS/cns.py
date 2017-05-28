import numpy
import scipy
import scipy.stats


class CNS:
    def __init__(self):
        print "init"

    def intra_coupling(self, attr, data):
        attr_list = data[:, attr].tolist()

        # print data.shape[0] # row
        # print data.shape[1] # col

        myset = set(attr_list)
        intra_matrix = numpy.zeros((len(myset), len(myset)))

        i = 0
        j = 0
        for x in myset:
            for y in myset:
                g_x = attr_list.count(x)
                g_y = attr_list.count(y)
                intra = 1.0 * (g_x * g_y) / (g_x + g_y + (g_x * g_y))
                intra_matrix[i, j] = intra
                j += 1
            j = 0
            i += 1

        return intra_matrix

    def intra_coupling(self, x, y, attr, data):
        attr_list = data[:, attr].tolist()
        g_x = attr_list.count(x)
        g_y = attr_list.count(y)
        intra = 1.0 * (g_x * g_y) / (g_x + g_y + (g_x * g_y))

        return intra

    def g(self, x, attr, data):
        attr_list = data[:, attr]
        ii = numpy.where(attr_list == x)[0]

        return ii

    def g_star(self, val_list, attr, data):
        ii = numpy.array([])
        for i in val_list:
            ii = numpy.union1d(ii, self.g(i, attr, data))
        return ii

    def icp(self, k, j, w, x, data):
        g_star_w = self.g_star(w, k, data)
        g_x = self.g(x, j, data)
        ii = 1.0*len(numpy.intersect1d(g_star_w, g_x)) / len(g_x)
        return ii

    def irsu(self, x, y, k, j, data):
        attr_list = data[:, k].tolist()

        myset = set(attr_list)
        sum_val = 0
        for i in myset:
            icp_x = self.icp(k, j, [i], x, data)
            icp_y = self.icp(k, j, [i], y, data)
            sum_val += max(icp_x, icp_y)
        result = 2 - sum_val
        return result

    def inter_coupling(self, j, x, y, data):
        para_alpha = 1.0/(data.shape[1]-1)
        result = 0
        for k in xrange(0, data.shape[1]):
            if (k != j):
                result += para_alpha * self.irsu(x, y, k, j, data)
        return result

    def cos(self, u1, u2, data):
        result = 0
        for j in xrange(0, data.shape[1]):
            x = data[u1][j]
            y = data[u2][j]
            intra = self.intra_coupling(x, y, j, data)
            inter = self.inter_coupling(j, x, y, data)
            a = intra*inter
            result += intra*inter
        return result/data.shape[1]

    def cos_matrix(self, data):
        row_count = data.shape[0]

        cos_matrix = numpy.zeros((row_count, row_count))

        for i in xrange(0, row_count):
            for j in xrange(0, row_count):
                cos_matrix[i][j] = "{0:.2f}".format(self.cos(i, j, data))
        return cos_matrix

    def load_data(self):
        f = open('../data/LazegaLawyers/ELattr36.dat')
        data = numpy.loadtxt(f)
        data1 = numpy.delete(data, numpy.s_[0], 1)
        data2 = numpy.delete(data1, numpy.s_[0, 1, 5], 1)
        print data2
        return data2 #.astype(numpy.int)

    def sigmoid(self, x):
        a = 1.0 / (1.0 + numpy.exp(-x))
        return a

    def convert_range(self, matrix, new_min, new_max):
        old_min = matrix.min()
        old_max = matrix.max()
        for i in xrange(0, matrix.shape[0]):
            for j in xrange(0, matrix.shape[1]):
                matrix[i][j] = (1.0*(matrix[i][j] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        return matrix

    def run(self):

        data = numpy.array([[1.0, 1.0, 1.0],
                            [2.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0],
                            [3.0, 3.0, 2.0],
                            [4.0, 3.0, 3.0],
                            [4.0, 2.0, 3.0]])

        #data = self.load_data()
        #intra = self.intra_coupling(6, data)
        #intra = self.g(1, 2, data)
        #intra = self.g_star([2,1], 6, data)
        #intra = self.irsu(1, 2, 1, 0, data)
        #intra = self.inter_coupling(1, 1, 2, data)
        #intra = self.cos(1, 2, data)
        intra = self.cos_matrix(data)
        v = self.convert_range(intra, -1.0, 1.0)
        print v
        #numpy.savetxt("../out/out_lazega36.csv", intra, delimiter=" ", fmt='%1.2f')
        #print intra

if __name__ == '__main__':
    cos = CNS()
    cos.run()
    #data = cos.load_data()

