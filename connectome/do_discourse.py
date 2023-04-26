from wikipedia2vec import Wikipedia2Vec
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import csv
import random as rand

import scipy
from scipy import stats

from easyesn.optimizers import GradientOptimizer
from easyesn import PredictionESN
from easyesn.optimizers import GridSearchOptimizer
from easyesn import helper as hlp

def run_metusalem(subject, bundle_name):
    
    vectorDim = 100

    numNode = 1000

    train='C:/Users/PeterDell/Google Drive/GoogleWIP/Projects/TemporalTopography/python_transfer/in_small.npy'
    test='C:/Users/PeterDell/Google Drive/GoogleWIP/Projects/TemporalTopography/python_transfer/out_small.npy'

    inputDataTraining = np.load(train)
    outputDataTraining = np.load(test)

    washout = inputDataTraining[:100,:]
    bufferwords=[]

    def cos_sim(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    wiki2vec = Wikipedia2Vec.load('C:/Users/PeterDell/Google Drive/GoogleWIP/People/Uchida/PaperMethods/UchidaPrograms/enwiki_20180420_100d.pkl')


    N = 72

    reps = 50

    A = np.empty(N)
    B = np.empty(N)
    C = np.empty(N)
    D = np.empty(N)
    E = np.empty(N)
    F = np.empty(N)

    Areps = np.empty(reps)
    Breps = np.empty(reps)
    Creps = np.empty(reps)
    Dreps = np.empty(reps)
    Ereps = np.empty(reps)
    Freps = np.empty(reps)

    f = open('metusalem2012_experiment_50_topo.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')

    # here is the loop on reservoir instances
    for instances in range(reps):

      # training the reservoir

      np.random.seed(instances)
      
      # the first set was with leak rate 0.05, now trying 0.2
      print('Start reservoir training', instances)
      esn = PredictionESN(n_input=vectorDim, n_output=vectorDim, n_reservoir=numNode, leakingRate=0.05, regressionParameters=[1e-2], solver="lsqr", feedback=False)
      esn.fit(inputDataTraining, outputDataTraining, transientTime="Auto", verbose=1)

      print('Reservoir trainging done')

      # here is where we make the connectome changes

      # from PT http://localhost:8890/notebooks/Pathway_subtractions_for_GIT_clean.ipynb
      # get full brain connectome and normalize it
      #subject=100307#100206#100307#
      weightsf='C:/data/for_Peter/%s/T1w/Diffusion/weights_Schaefer2018_1000Parcels_17Networks_15M.txt' % (subject)
      connectome_weights = np.genfromtxt(str(weightsf))
      connectome_weights = connectome_weights[:1000,:1000] # remove subcortical areas
      connectome_weights = np.log(connectome_weights+1)
      connectome_weights_max = connectome_weights.max()
      connectome_weights /= connectome_weights_max


      # modify input weights of the reservoir such that only certain brain regions receive an input
      # which areas are those ???
      esn._WInput[150:500,:101] = 0
      esn._WInput[650:1000,:101] = 0
      esn._WInput *= 5 # why multiply by 5 ?

      # use the connectome to change the weights (_W) of the reservoir
      n_c_gain  = 2
      esn._W = connectome_weights * esn._W * n_c_gain 
      esn_w_count = np.count_nonzero(esn._W)
      
      
      
      
      print("**************************************** testing *************")
      print(" ****************** Connectome esn._W non-zeros: ",esn_w_count)
      """ this is where we were testing random removal of 4 percent, same as MLF
      for i in range (1000):
        for j in range(1000):
            if(rand.random())<0.04:
                esn._W[i,j]=0
      esn_w_minus_count = np.count_nonzero(esn._W)
      print("**************  Connectome minus 3 percent esn._W non-zeros: ",esn_w_minus_count)
      print("percentage: ", 1-(esn_w_minus_count/esn_w_count))
      """
      
      #bundle_name="MLF_left"
      #bundle_name="IFO_left"
      if bundle_name != "None":
        
        print("removing", bundle_name)
        # get bundle connectome
        bundlef = 'C:/data/for_Peter/%s/T1w/Diffusion/TractSeg/tck_segmentations/bundle_connectome_%s_weights.txt' % (subject,bundle_name)
        bundle_weights = np.genfromtxt(str(bundlef))
        bundle_weights = np.log(bundle_weights[:1000,:1000]+1)/connectome_weights_max
        
        # now run the time constant exercise on the lesioned weights using the tracts that were selected
        truncated_weights = connectome_weights.copy() #- bundle_weights
        truncated_weights[bundle_weights > 0] = 0
        #esn._W = truncated_weights * esn._W * n_c_gain 
        
        #after testing, change it to:
        esn._W[bundle_weights > 0] = 0
        esn_w_minus_count = np.count_nonzero(esn._W)
        print("**************  Connectome minus bundle esn._W non-zeros: ",esn_w_minus_count)
        print("percentage: ", 1-(esn_w_minus_count/esn_w_count))
        
        
        
      # here is the loop on scenarios
      for i in range(N):
        #print('\n############### ' + str(i + 1) + ' ###############')

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


        """
        print('Data:')
        print('target_word_1: %s' % target_word_1)
        print('target_word_2: %s' % target_word_2)
        print('target_word_3: %s' % target_word_3)

        print('discourse_words_1:')
        print(discourse_words_1)
        print('discourse_words_1and2:')
        print(discourse_words_1and2)
        """
        

        inputDataTesting = np.empty((0,vectorDim))
        #print(inputDataTesting.shape)
        for num in range(len(bufferwords)):
            inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(bufferwords[num])]), axis=0)
        #inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector("noise")]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_1)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_1)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_1)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_1)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_1)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_1)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_1)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_1)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_1)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_1)]), axis=0)

        #print(inputDataTesting.shape)
        reservoirStatesBuffer = np.empty((0,numNode))
        prediction,reservoirStatesBuffer  = esn.predict(washout)
        prediction,reservoirStatesBuffer  = esn.predict(inputDataTesting)
        reservoirStatesBuffer = reservoirStatesBuffer.T
        reservoirStatesBuffer = reservoirStatesBuffer[:,101:]
        target_word_1_vector = reservoirStatesBuffer[len(reservoirStatesBuffer)-1]    
        # end of creating discourse vector by reservoir

        


        inputDataTesting = np.empty((0,vectorDim))
        #print(inputDataTesting.shape)
        for num in range(len(bufferwords)):
            inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(bufferwords[num])]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_3)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_3)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_3)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_3)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_3)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_3)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_3)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_3)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_3)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_3)]), axis=0)
        #print(inputDataTesting.shape)
        reservoirStatesBuffer = np.empty((0,numNode))
        prediction,reservoirStatesBuffer  = esn.predict(washout)
        prediction,reservoirStatesBuffer  = esn.predict(inputDataTesting)
        reservoirStatesBuffer = reservoirStatesBuffer.T
        reservoirStatesBuffer = reservoirStatesBuffer[:,101:]
        target_word_3_vector = reservoirStatesBuffer[len(reservoirStatesBuffer)-1]    

        inputDataTesting = np.empty((0,vectorDim))
        #print(inputDataTesting.shape)
        for num in range(len(bufferwords)):
            inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(bufferwords[num])]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_2)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_2)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_2)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_2)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_2)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_2)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_2)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_2)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_2)]), axis=0)
        inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(target_word_2)]), axis=0)
        #print(inputDataTesting.shape)
        reservoirStatesBuffer = np.empty((0,numNode))
        prediction,reservoirStatesBuffer  = esn.predict(washout)
        prediction,reservoirStatesBuffer  = esn.predict(inputDataTesting)
        reservoirStatesBuffer = reservoirStatesBuffer.T
        reservoirStatesBuffer = reservoirStatesBuffer[:,101:]
        target_word_2_vector = reservoirStatesBuffer[len(reservoirStatesBuffer)-1]    

        '''
        fig, ax = plt.subplots()
        t = np.linspace(1, 2, 2)
        '''

        trajectory_word_1 = np.array([])
        trajectory_word_2 = np.array([])
        trajectory_word_3 = np.array([])

        #print('\nStep1: ')
     

        #now we want to create the dicourse_vector_1 using the reservoir

        inputDataTesting = np.empty((0,vectorDim))
        #print(inputDataTesting.shape)

        for num in range(len(discourse_words_1)):
            inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(discourse_words_1[num])]), axis=0)
        #print(inputDataTesting.shape)

        #prediction = esn.predict(inputDataTesting)

        reservoirStatesBuffer = np.empty((0,numNode))
        prediction,reservoirStatesBuffer  = esn.predict(washout)
        prediction,reservoirStatesBuffer  = esn.predict(inputDataTesting)
        #print(prediction.shape)
        #print(len(prediction))
        discourse_vector_old = prediction[len(prediction)-1]
        # we are going to measure the cosine differences, but now we want to do it 
        # with the reservoir internal states instead of the predicted output
        reservoirStatesBuffer = reservoirStatesBuffer.T
        #print("rereservoirStatesBuffer.shape before truncating",reservoirStatesBuffer.shape)
        reservoirStatesBuffer = reservoirStatesBuffer[:,101:]
        #print("reservoirStatesBuffer.shape before truncating",reservoirStatesBuffer.shape)
        discourse_vector_1 = reservoirStatesBuffer[len(reservoirStatesBuffer)-1]    
        # end of creating discourse vector by reservoir
        
        """
        print('cos(target_word_1_vector, target_word_2_vector)=%f' % (cos_sim(target_word_1_vector, target_word_2_vector)))
        print('cos(target_word_1_vector, target_word_3_vector)=%f' % (cos_sim(target_word_1_vector, target_word_3_vector)))
        print('cos(target_word_2_vector, target_word_3_vector)=%f' % (cos_sim(target_word_2_vector, target_word_3_vector)))

        # end of creating discourse vector by reservoir

        print('cos(discourse_vector_1, %s)=%f' % (target_word_1, cos_sim(discourse_vector_1, target_word_1_vector)))
        print('cos(discourse_vector_1, %s)=%f' % (target_word_2, cos_sim(discourse_vector_1, target_word_2_vector)))
        print('cos(discourse_vector_1, %s)=%f' % (target_word_3, cos_sim(discourse_vector_1, target_word_3_vector)))
        """
        trajectory_word_1 = np.append(trajectory_word_1, cos_sim(discourse_vector_1, target_word_1_vector))
        trajectory_word_2 = np.append(trajectory_word_2, cos_sim(discourse_vector_1, target_word_2_vector))
        trajectory_word_3 = np.append(trajectory_word_3, cos_sim(discourse_vector_1, target_word_3_vector))

        A[i] = cos_sim(discourse_vector_1, target_word_1_vector)
        B[i] = cos_sim(discourse_vector_1, target_word_2_vector)
        C[i] = cos_sim(discourse_vector_1, target_word_3_vector)


        #print('\nStep2: ')


        #now we want to create the dicourse_vector_1 using the reservoir

        inputDataTesting = np.empty((0,vectorDim))
        #print(inputDataTesting.shape)

        for num in range(len(discourse_words_1and2)):
            inputDataTesting = np.append(inputDataTesting, np.array([wiki2vec.get_word_vector(discourse_words_1and2[num])]), axis=0)
        #print(inputDataTesting.shape)

        #prediction = esn.predict(inputDataTesting)
        prediction,reservoirStatesBuffer  = esn.predict(washout)
        prediction,reservoirStatesBuffer = esn.predict(inputDataTesting)
        #print(prediction)
        #print(prediction.shape)
        #print(len(prediction))
        #discourse_vector_1and2 = prediction[len(prediction)-1]
        # we are going to measure the cosine differences, but now we want to do it 
        # with the reservoir internal states instead of the predicted output
        reservoirStatesBuffer = reservoirStatesBuffer.T
        #print("rereservoirStatesBuffer.shape before truncating",reservoirStatesBuffer.shape)
        reservoirStatesBuffer = reservoirStatesBuffer[:,101:]
        #print("reservoirStatesBuffer.shape before truncating",reservoirStatesBuffer.shape)
        discourse_vector_1and2 = reservoirStatesBuffer[len(reservoirStatesBuffer)-1]    

        """
        # end of creating discourse vector by reservoir
        print('cos(discourse_vector_1and2, %s)=%f' % (target_word_1, cos_sim(discourse_vector_1and2, target_word_1_vector)))
        print('cos(discourse_vector_1and2, %s)=%f' % (target_word_2, cos_sim(discourse_vector_1and2, target_word_2_vector)))
        print('cos(discourse_vector_1and2, %s)=%f' % (target_word_3, cos_sim(discourse_vector_1and2, target_word_3_vector)))
        """
        writer.writerow([cos_sim(discourse_vector_1, target_word_1_vector), cos_sim(discourse_vector_1, target_word_2_vector), cos_sim(discourse_vector_1, target_word_3_vector), cos_sim(discourse_vector_1and2, target_word_1_vector), cos_sim(discourse_vector_1and2, target_word_2_vector), cos_sim(discourse_vector_1and2, target_word_3_vector)])

        D[i] = cos_sim(discourse_vector_1and2, target_word_1_vector)
        E[i] = cos_sim(discourse_vector_1and2, target_word_2_vector)
        F[i] = cos_sim(discourse_vector_1and2, target_word_3_vector)

        trajectory_word_1 = np.append(trajectory_word_1, cos_sim(discourse_vector_1and2, target_word_1_vector))
        trajectory_word_2 = np.append(trajectory_word_2, cos_sim(discourse_vector_1and2, target_word_2_vector))
        trajectory_word_3 = np.append(trajectory_word_3, cos_sim(discourse_vector_1and2, target_word_3_vector))

     

      f.close()
      Areps[instances] = np.mean(A)
      Breps[instances] = np.mean(B)
      Creps[instances] = np.mean(C)
      Dreps[instances] = np.mean(D)
      Ereps[instances] = np.mean(E)
      Freps[instances] = np.mean(F)


    data_to_plot = [Areps, Breps, Creps, Dreps, Ereps, Freps]

    #np.save('Exp3C-50-reservoir-data_topo_connectome_backup.npy', data_to_plot) 
    #filename = 'connectome_inference/subject_%s_path_%s.npy' % (subject,bundle_name)
    filename = 'connectome_inference_check/subject_%s_path_%s_new_esb.npy' % (subject,bundle_name)
    np.save(filename, data_to_plot)

    #print("saved: ", filename)


