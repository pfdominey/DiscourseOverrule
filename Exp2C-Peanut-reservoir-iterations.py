from easyesn.optimizers import GradientOptimizer
from easyesn import PredictionESN
from easyesn.optimizers import GridSearchOptimizer
from easyesn import helper as hlp
import numpy as np
from wikipedia2vec import Wikipedia2Vec

import matplotlib.pyplot as plt

import scipy
from scipy import stats

vectorDim = 100

numNode = 100

inputDataTraining = np.load('./trainingData_averaging/inputDataTraining_4k_average.npy')
outputDataTraining = np.load('./trainingData_averaging/outputDataTraining_4k_average.npy')

wiki2vec = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# number of repetitions of the reservoir

reps = 50

# output data structures

love_before = np.empty(reps)
salted_before = np.empty(reps)
love_after = np.empty(reps)
salted_after = np.empty(reps)

#start the loop on iterations
# here is the loop on reservoir instances
for instances in range(reps):


    np.random.seed(instances)

    print('start training', instances)
    esn = PredictionESN(n_input=vectorDim, n_output=vectorDim, n_reservoir=numNode, leakingRate=0.2, regressionParameters=[1e-2], solver="lsqr", feedback=False)
    esn.fit(inputDataTraining, outputDataTraining, transientTime="Auto", verbose=1)
    print('end training')


    discourse_words = ['peanut', 'neutral','woman', 'saw', 'dancing', 'peanut', 'big', 'smile', 'his', 'face', 'peanut', 'singing', 'girl', 'just', 'met', 'judging', 'song', 'peanut', 'totally', 'crazy', 'her', 'woman', 'thought', 'really', 'cute', 'see', 'peanut', 'singing', 'dancing', 'peanut']
    #discourse_words = ['peanuts', 'snacks', 'bowl', 'bar', 'people', 'eating', 'chips', 'drinking', 'beer', 'watching', 'football', 'game', 'woman', 'saw', 'dancing', 'peanut', 'big', 'smile', 'his', 'face', 'peanut', 'singing', 'girl', 'just', 'met', 'judging', 'song', 'peanut', 'totally', 'crazy', 'her', 'woman', 'thought', 'really', 'cute', 'see', 'peanut', 'singing', 'dancing', 'peanut']
    #print(discourse_words)

    inputDataTesting = np.empty((0,vectorDim))
    print(inputDataTesting.shape)

    for num in range(len(discourse_words)):
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(discourse_words[num])]), axis=0)
    print(inputDataTesting.shape)



    prediction = esn.predict(inputDataTesting)
    #print(prediction)


    # generate the discourse trajectory for consistent and iconsistent
    consistentOutputDataTesting = np.empty((0,vectorDim))
    inconsistentOutputDataTesting = np.empty((0,vectorDim))
    print(consistentOutputDataTesting.shape)
    print(inconsistentOutputDataTesting.shape)

    for num in range(len(discourse_words) - 1):
        consistentOutputDataTesting = np.append(consistentOutputDataTesting, np.array([wiki2vec.get_word_vector(discourse_words[num + 1])]), axis=0)
        inconsistentOutputDataTesting = np.append(inconsistentOutputDataTesting, np.array([wiki2vec.get_word_vector(discourse_words[num + 1])]), axis=0)
    consistentOutputDataTesting = np.append(consistentOutputDataTesting, np.array([wiki2vec.get_word_vector('love')]), axis=0)
    inconsistentOutputDataTesting = np.append(inconsistentOutputDataTesting, np.array([wiki2vec.get_word_vector('salted')]), axis=0)
    print(consistentOutputDataTesting.shape)
    print(inconsistentOutputDataTesting.shape)


    consistent_trajectory = np.array([])
    inconsistent_trajectory = np.array([])

    consistent_vector = wiki2vec.get_word_vector('love')
    inconsistent_vector = wiki2vec.get_word_vector('salted')

    for num in range(len(prediction)):
        consistent_trajectory = np.append(consistent_trajectory, cos_sim(consistent_vector, prediction[num]))
        inconsistent_trajectory = np.append(inconsistent_trajectory, cos_sim(inconsistent_vector, prediction[num]))
        #consistent_trajectory = np.append(consistent_trajectory, cos_sim(consistentOutputDataTesting[num], prediction[num]))
        #inconsistent_trajectory = np.append(inconsistent_trajectory, cos_sim(inconsistentOutputDataTesting[num], prediction[num]))
    print(inconsistent_trajectory)
    print(consistent_trajectory)

    # accumulate repeated values

    love_before[instances]= consistent_trajectory[0]
    salted_before[instances] = inconsistent_trajectory[0]
    love_after[instances] = consistent_trajectory[num]
    salted_after[instances] = inconsistent_trajectory[num]


    # plot the graph

    fig, ax = plt.subplots()
    t = np.linspace(1, len(discourse_words), len(discourse_words))

    ax.set_xlabel('number of word')
    ax.set_ylabel('cosine similarity')
    ax.set_title(r'cosine similarity trajectory')
    ax.set_xlim([1,len(discourse_words)])
    ax.set_ylim([0, 1])

    ax.plot(t, inconsistent_trajectory, color="blue", label="inconsistent")
    ax.plot(t, consistent_trajectory, color="red", label="consistent")

    ax.legend(loc=0)
    fig.tight_layout()
    plt.savefig('./peanut-results/'+str(instances + 1)+'.png')

# stats

print('t-test: for love_before vs salted_before: ', stats.ttest_ind(love_before,salted_before))
print('t-test: for love_after vs salted_after: ', stats.ttest_ind(love_after,salted_after))



# now plot the cumuated results

data_to_plot = [love_before, salted_before, love_after, salted_after]
#print(data_to_plot)

np.save('Exp2C-reservoir-data.npy', data_to_plot) 

# Create a figure instance
#fig = plt.figure(1, figsize=(9, 6))
fig2 = plt.figure(1)
# Create an axes instance
ax = fig2.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data_to_plot)

ax.set_xticklabels(['love_before', 'salted_before', 'love_after', 'salted_after'])
fig2.tight_layout()
plt.savefig('./peanut-results/peanut-res-reps.png')





