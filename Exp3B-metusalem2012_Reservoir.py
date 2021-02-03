from wikipedia2vec import Wikipedia2Vec
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import csv

import scipy
from scipy import stats

from easyesn.optimizers import GradientOptimizer
from easyesn import PredictionESN
from easyesn.optimizers import GridSearchOptimizer
from easyesn import helper as hlp


vectorDim = 100



def cos_sim(v1, v2):
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

wiki2vec = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')

# training the reservoir

np.random.seed(2)
vectorDim = 100

numNode = 100

inputDataTraining = np.load('./trainingData_averaging/inputDataTraining_4k_average.npy')
outputDataTraining = np.load('./trainingData_averaging/outputDataTraining_4k_average.npy')

print('Start reservoir training')
esn = PredictionESN(n_input=vectorDim, n_output=vectorDim, n_reservoir=numNode, leakingRate=0.2, regressionParameters=[1e-2], solver="lsqr", feedback=False)
esn.fit(inputDataTraining, outputDataTraining, transientTime="Auto", verbose=1)

print('Reservoir trainging done')


N = 72

A = np.empty(N)
B = np.empty(N)
C = np.empty(N)
D = np.empty(N)
E = np.empty(N)
F = np.empty(N)

vectors = np.empty([N,vectorDim])


f = open('metusalem2012_experiment_reservoir.csv', 'w')
writer = csv.writer(f, lineterminator='\n')

for i in range(N):
    print('\n############### ' + str(i + 1) + ' ###############')

    # read txt file for data
    f = open('./data_metusalem2012/'+str(i + 1)+'.txt', 'r')
    list = f.readlines()
    discourse_words_1 = list[1].split()
    discourse_words_2 = list[0].split()
    discourse_words_1and2 = discourse_words_2 + discourse_words_1
    target_word_1 = list[2].lower()
    target_word_2 = list[3].lower()
    target_word_3 = list[4].lower()
    f.close()

    # large capital -> small capital
    discourse_words_1 = [s.replace(s, s.lower()) for s in discourse_words_1]
    discourse_words_1and2 = [s.replace(s, s.lower()) for s in discourse_words_1and2]

    # remove '.' and ',' from word list
    discourse_words_1 = [s.replace('.', '') for s in discourse_words_1]
    discourse_words_1and2 = [s.replace('.', '') for s in discourse_words_1and2]
    discourse_words_1 = [s.replace(',', '') for s in discourse_words_1]
    discourse_words_1and2 = [s.replace(',', '') for s in discourse_words_1and2]

    # remove stop words from word list
    stop_words = stopwords.words('english')
    #print(stop_words)
    for stop_word in stop_words:
        while stop_word in discourse_words_1 :
            discourse_words_1.remove(stop_word)
            
        while stop_word in discourse_words_1and2 :
            discourse_words_1and2.remove(stop_word)
            
    # remove "'s" and "'" and "-" and "'d" and "'ll" and "'ve" and "re" from word list
    discourse_words_1 = [s.replace("'s", '') for s in discourse_words_1]
    discourse_words_1and2 = [s.replace("'s", '') for s in discourse_words_1and2]
    discourse_words_1 = [s.replace("'", '') for s in discourse_words_1]
    discourse_words_1and2 = [s.replace("'", '') for s in discourse_words_1and2]
    discourse_words_1 = [s.replace("-", '') for s in discourse_words_1]
    discourse_words_1and2 = [s.replace("-", '') for s in discourse_words_1and2]
    discourse_words_1 = [s.replace("'d", '') for s in discourse_words_1]
    discourse_words_1and2 = [s.replace("'d", '') for s in discourse_words_1and2]
    discourse_words_1 = [s.replace("'ll", '') for s in discourse_words_1]
    discourse_words_1and2 = [s.replace("'ll", '') for s in discourse_words_1and2]
    discourse_words_1 = [s.replace("'ve", '') for s in discourse_words_1]
    discourse_words_1and2 = [s.replace("'ve", '') for s in discourse_words_1and2]
    discourse_words_1 = [s.replace("'re", '') for s in discourse_words_1]
    discourse_words_1and2 = [s.replace("'re", '') for s in discourse_words_1and2]

    # replace '\n' from target words
    target_word_1 = target_word_1.replace('\n', '')
    target_word_2 = target_word_2.replace('\n', '')
    target_word_3 = target_word_3.replace('\n', '')


    print('Data:')
    print('target_word_1: %s' % target_word_1)
    print('target_word_2: %s' % target_word_2)
    print('target_word_3: %s' % target_word_3)

    print('discourse_words_1:')
    print(discourse_words_1)
    print('discourse_words_1and2:')
    print(discourse_words_1and2)


    target_word_1_vector = wiki2vec.get_word_vector(target_word_1)
    target_word_2_vector = wiki2vec.get_word_vector(target_word_2)
    target_word_3_vector = wiki2vec.get_word_vector(target_word_3)

    '''
    fig, ax = plt.subplots()
    t = np.linspace(1, 2, 2)
    '''

    trajectory_word_1 = np.array([])
    trajectory_word_2 = np.array([])
    trajectory_word_3 = np.array([])

    print('\nStep1: ')
 
    # here we create discourse_vector_1 by averaging word2vecs
    '''
    for num in range(len(discourse_words_1)):
        #print(discourse_words_1[num])
        if num == 0:
            discourse_vector_1 = wiki2vec.get_word_vector(discourse_words_1[num])
        else:
            discourse_vector_1 = (num * discourse_vector_1 + wiki2vec.get_word_vector(discourse_words_1[num])) / (num + 1)
    '''
    # end of creating discourse vector by averaging

    #now we want to create the dicourse_vector_1 using the reservoir

    inputDataTesting = np.empty((0,vectorDim))
    print(inputDataTesting.shape)

    for num in range(len(discourse_words_1)):
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(discourse_words_1[num])]), axis=0)
    print(inputDataTesting.shape)

    prediction = esn.predict(inputDataTesting)
    #print(prediction)
    print(prediction.shape)
    print(len(prediction))
    discourse_vector_1 = prediction[len(prediction)-1]

    # end of creating discourse vector by reservoir

    print('cos(discourse_vector_1, %s)=%f' % (target_word_1, cos_sim(discourse_vector_1, target_word_1_vector)))
    print('cos(discourse_vector_1, %s)=%f' % (target_word_2, cos_sim(discourse_vector_1, target_word_2_vector)))
    print('cos(discourse_vector_1, %s)=%f' % (target_word_3, cos_sim(discourse_vector_1, target_word_3_vector)))

    trajectory_word_1 = np.append(trajectory_word_1, cos_sim(discourse_vector_1, target_word_1_vector))
    trajectory_word_2 = np.append(trajectory_word_2, cos_sim(discourse_vector_1, target_word_2_vector))
    trajectory_word_3 = np.append(trajectory_word_3, cos_sim(discourse_vector_1, target_word_3_vector))

    A[i] = cos_sim(discourse_vector_1, target_word_1_vector)
    B[i] = cos_sim(discourse_vector_1, target_word_2_vector)
    C[i] = cos_sim(discourse_vector_1, target_word_3_vector)


    print('\nStep2: ')

    '''
    for num in range(len(discourse_words_1and2)):
        #print(discourse_words_1and2[num])
        if num == 0:
            discourse_vector_1and2 = wiki2vec.get_word_vector(discourse_words_1and2[num])
        else:
            discourse_vector_1and2 = (num * discourse_vector_1and2 + wiki2vec.get_word_vector(discourse_words_1and2[num])) / (num + 1)
    '''

    #now we want to create the dicourse_vector_1 using the reservoir

    inputDataTesting = np.empty((0,vectorDim))
    print(inputDataTesting.shape)

    for num in range(len(discourse_words_1and2)):
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(discourse_words_1and2[num])]), axis=0)
    print(inputDataTesting.shape)

    prediction = esn.predict(inputDataTesting)
    #print(prediction)
    print(prediction.shape)
    print(len(prediction))
    discourse_vector_1and2 = prediction[len(prediction)-1]

    # end of creating discourse vector by reservoir

    vectors[i] = discourse_vector_1and2
    
    print('cos(discourse_vector_1and2, %s)=%f' % (target_word_1, cos_sim(discourse_vector_1and2, target_word_1_vector)))
    print('cos(discourse_vector_1and2, %s)=%f' % (target_word_2, cos_sim(discourse_vector_1and2, target_word_2_vector)))
    print('cos(discourse_vector_1and2, %s)=%f' % (target_word_3, cos_sim(discourse_vector_1and2, target_word_3_vector)))
    
    writer.writerow([cos_sim(discourse_vector_1, target_word_1_vector), cos_sim(discourse_vector_1, target_word_2_vector), cos_sim(discourse_vector_1, target_word_3_vector), cos_sim(discourse_vector_1and2, target_word_1_vector), cos_sim(discourse_vector_1and2, target_word_2_vector), cos_sim(discourse_vector_1and2, target_word_3_vector)])

    D[i] = cos_sim(discourse_vector_1and2, target_word_1_vector)
    E[i] = cos_sim(discourse_vector_1and2, target_word_2_vector)
    F[i] = cos_sim(discourse_vector_1and2, target_word_3_vector)

    trajectory_word_1 = np.append(trajectory_word_1, cos_sim(discourse_vector_1and2, target_word_1_vector))
    trajectory_word_2 = np.append(trajectory_word_2, cos_sim(discourse_vector_1and2, target_word_2_vector))
    trajectory_word_3 = np.append(trajectory_word_3, cos_sim(discourse_vector_1and2, target_word_3_vector))

    data_to_plot = [A[0:i], B[0:i], C[0:i], D[0:i], E[0:i], F[0:i]]

    '''
    # Create a figure instance
    #fig = plt.figure(1, figsize=(9, 6))
    fig = plt.figure(1)
    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot)
    plt.ylim(0.25, 0.75)

    ax.set_xticklabels(['Sent-Expect', 'Sent-Relat', 'Sent-Unrel', 'Disc-Expect', 'Disc-Relat', 'Disc-Unrel'])
    fig.tight_layout()
    plt.savefig('./figures_metusalem_single_reservoir/'+str(i)+'.png')
    plt.close()
    '''


f.close()



data_to_plot = [A, B, C, D, E, F]

np.save('Exp3B-reservoir-data.npy', data_to_plot) 

print(data_to_plot)

print('independant samples')
print('t-test: for A vs B: ', stats.ttest_ind(A,B))
print('t-test: for B vs C: ', stats.ttest_ind(B,C))
print('t-test: for D vs E: ', stats.ttest_ind(D,E))
print('t-test: for E vs F: ', stats.ttest_ind(E,F))

print('dependant samples')
print('t-test: for A vs B: ', stats.ttest_rel(A,B))
print('t-test: for B vs C: ', stats.ttest_rel(B,C))
print('t-test: for D vs E: ', stats.ttest_rel(D,E))
print('t-test: for E vs F: ', stats.ttest_rel(E,F))


