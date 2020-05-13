from wikipedia2vec import Wikipedia2Vec
#import gensim
import numpy as np
import matplotlib.pyplot as plt

def cos_sim(v1, v2):
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

wiki2vec = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')

subject_word = 'peanut'
target_word_false = 'salted'
target_word_true = 'love'
discourse_words_with_subject = ['peanut', 'neutral', 'woman', 'saw', 'dancing', 'peanut', 'big', 'smile', 'his', 'face', 'peanut', 'singing', 'girl', 'just', 'met', 'judging', 'song', 'peanut', 'totally', 'crazy', 'her', 'woman', 'thought', 'really', 'cute', 'see', 'peanut', 'singing', 'dancing', 'peanut']
fig_name = 'u_script7-trajectory.png'
'''


#7-new
subject_word = 'peanut'
target_word_false = 'salted'
target_word_true = 'love'
discourse_words_with_subject = ['peanuts', 'snacks', 'bowl', 'bar', 'people', 'eating', 'chips', 'drinking', 'beer', 'watching', 'football', 'game', 'woman', 'saw', 'dancing', 'peanut', 'big', 'smile', 'his', 'face', 'peanut', 'singing', 'girl', 'just', 'met', 'judging', 'song', 'peanut', 'totally', 'crazy', 'her', 'woman', 'thought', 'really', 'cute', 'see', 'peanut', 'singing', 'dancing', 'peanut']
fig_name = 'u_script7_new-trajectory.png'
'''


print('Data:')
print('subject_word: %s' % subject_word)
print('target_word_false: %s' % target_word_false)
print('target_word_true: %s' % target_word_true)

print('discourse_words_with_subject:')
print(discourse_words_with_subject)
#print('discourse_words_without_subject:')
#print(discourse_words_without_subject)

#print(model.similarity(subject, target_word_false))
#print(model.similarity(subject, target_word_true))

subject_vector = wiki2vec.get_word_vector(subject_word)
target_vector_false = wiki2vec.get_word_vector(target_word_false)
target_vector_true = wiki2vec.get_word_vector(target_word_true)

print('\nStep1: cosine similarity between subject and target words')
print('cos(%s, %s)=%f' % (subject_word, target_word_false, cos_sim(subject_vector, target_vector_false)))
print('cos(%s, %s)=%f' % (subject_word, target_word_true, cos_sim(subject_vector, target_vector_true)))



print('\nStep5: cosine similarity trajectory (discourse_words_with_subject)')

fig, ax = plt.subplots()

#print('len(discourse_words_with_subject)')
#print(len(discourse_words_with_subject))
t = np.linspace(1, len(discourse_words_with_subject), len(discourse_words_with_subject))
#print(t)

trajectory_with_false = np.array([])
trajectory_with_true = np.array([])



for num in range(len(discourse_words_with_subject)):
	#print(discourse_words_with_subject[num])
	if num == 0:
		discourse_vector4trajectory = wiki2vec.get_word_vector(discourse_words_with_subject[num])
	else:
		discourse_vector4trajectory = (num * discourse_vector4trajectory + wiki2vec.get_word_vector(discourse_words_with_subject[num])) / (num + 1)
	trajectory_with_false = np.append(trajectory_with_false, cos_sim(target_vector_false, discourse_vector4trajectory))
	trajectory_with_true = np.append(trajectory_with_true, cos_sim(target_vector_true, discourse_vector4trajectory))
print(trajectory_with_false)
print(trajectory_with_true)

ax.set_xlabel('Word Number in Discourse')
ax.set_ylabel('Semantic Relatedness')
ax.set_title(r'Semantic Relatedness Reversal*')
ax.set_xlim([1,len(discourse_words_with_subject)])
ax.set_ylim([0.3, 0.8])

ax.plot(t, trajectory_with_false, color="blue", label="Salted")
ax.plot(t, trajectory_with_true, color="red", label="Love")

ax.legend(loc=0)
fig.tight_layout()
plt.savefig('Peanut-Average-Trajectory.png')
plt.show()



