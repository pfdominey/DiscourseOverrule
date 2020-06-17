import spacy 

# To create a model from a custom word_vector text file, run in command line : 
# python -m spacy init-model <language> <mode_name> --vectors-loc <word_vector_file_path>
# where 
#    <language> is the spacy shortname for the word vector language (en for english, ja for japanese)
#    <model_name> is the folder name where spacy will create the model. The folder does need to exist already
#    <word_vector_file_path> is the path to the word vector file
# See https://spacy.io/usage/vectors-similarity#converting for more information
#
# After spacy has created the model, you can create the corresponding WordVectors object by using : 
#    wv = WordVectors(<model_name>)
# And compute the word vector of a word by
#    word_vector = wv.get_word_vector(<word>)

# Example:
# In bash:
#     python -m spacy init-model en glove_6B_50d_twitter/ --vectors-loc glove.6B.50d.txt
# In python console:
#     wv = WordVectors('glove_6B_50d_twitter')
#     vector wv.get_word_vector('dog')

class WordVectors:
    def __init__(self, model_name, unk_word='raise_error'):
        self.unk_word = unk_word
        try:
            self.nlp = spacy.load(model_name)      
        except OSError as e:
            print(e)
            raise TypeError('You first need to create the spacy model using the "python -m spacy init-model..." command ')
            
    def get_word_vector(self, word):
        if isinstance(word, str):
            vector = nlp(word).vector
            if self.unk_word == 'raise_error' and not np.any(vector):
                    raise TypeError(f'The word {word} is not in the vocabulary, change attributes unk_word to None to ignore the error. A zero vector will be returned for unkown words')    
            return vector
        else:
            raise NotImplementedError('This function accepts only string as input')
