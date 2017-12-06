from stemmer import PorterStemmer
from WAKNN import WAKNN
import re
#
# port_stemmer = PorterStemmer()
#
# print(port_stemmer.stem("We're", 0, len("hard.") - 1))


knn = WAKNN(100, 10)

knn.load("reuters.csv")
knn.buildMatrix()
knn.normalize()
knn.initializeWeight()

# knn.weightedCosine(knn.documents[0], knn.documents[1], [1 for i in range(0, 2000)])
# neighbors = knn.knn(knn.documents[2])
# print(neighbors)
print(knn.training(0.5))
print(knn.majorityObj(0.5, [0.8, 1.0, 1.2000000000000002, 1.0, 1.0, 4.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
# print(knn.majorityObj(0.5, knn.weight))
# for neighbor in neighbors:
#     print(knn.labels[neighbor])
# knn.
# knn.classify(knn.documents[0], knn.documents, [1 for i in range(0, 2000)], 0)
# print([[1, 2,3 ], [1,3, 4]])
# print(re.sub(r'[^\w]', " ", "Showers continued throughout the week inthe Bahia cocoa zone, alleviating the drought since earlyJanuary and improving prospects for the coming temporao,although normal humidity levels have not been restored,Comissaria Smith said in its weekly review.    The dry period means the temporao will be late this year.    Arrivals for the week ended February 22 were 155,221 bagsof 60 kilos making a cumulative total for the season of 5.93mln against 5.81 at the same stage last year. Again it seemsthat cocoa delivered earlier on consignment was included in thearrivals figures.    Comissaria Smith said there is still some doubt as to howmuch old crop cocoa is still available as harvesting haspractically come to an end. With total Bahia crop "))