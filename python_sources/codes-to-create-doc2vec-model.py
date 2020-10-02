#######################PACKAGES
import os
import csv
import pandas  as pd
import nltk
import gensim
from nltk.corpus import stopwords
from gensim import corpora,models,similarities
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import  difflib
#os.chdir("F://python")   ######### CHANGE THE PATH AS PER YOUR LOCATION

##########################################################################################################
##################################### READING   CSV FILE #########################################

df=pd.read_csv("../dataset.csv", encoding = "ISO-8859-1")
df=df.dropna()   ###    Drop  if any Na values will  come

############################################################################################################

############# PREPROCESSING THE QUESTIONS IN OUR  CSV #####################################################################

for  i  in range(df.__len__()):
    questions=df['questions'][i].lower()
    stop_words = set(stopwords.words('english'))
    preprocess=[]
    from nltk.tokenize import TweetTokenizer
    tknzr = TweetTokenizer()
    preprocess.append(tknzr.tokenize(questions))
    print(preprocess)
    filtered_sentence = " ".join([w for w in preprocess[0] if not w in stop_words])
    print(filtered_sentence)
    tokens = nltk.wordpunct_tokenize(filtered_sentence)
    text = nltk.Text(tokens)
    words = " ".join([w.lower() for w in text if w.isalpha()])
    print(tokens,text,words)
    df['questions'][i]=words

#############################################################################################################

#############WRITE THE PREPROCESSED  CSV  FILE INTO LOCAL DRIVE  AS NAME  preprocessed.csv

df.to_csv("preprocessed.csv")

################ BUILDING A DOC2VEC MODEL ##################################################################

            # Various parameters
            #min_count = 1   # ignore all words with total frequency lower than this. You have to set this to 1, since the sentence labels only appear once.
            #workers = 11    # Number of threads to run in parallel
            #epocs = 20  ## number of itterations
            #window =2  ## the maximum distance between the current and predicted word within a sentence.
            # vector_size=100  dimensionality of the feature vectors in output


texts = df.to_dict('records')    #### DATAFRAME TO DICTIONERY FORMAT .
documents = [TaggedDocument(text['questions'].split(), [text['answers']])  for text in texts]   ## Pass the 'questions' & 'answers'
model = gensim.models.Doc2Vec(vector_size=100, window=2, min_count=1, workers=11,alpha=0.025, min_alpha=0.025, epochs=20)
model.build_vocab(documents)
model.train(documents, epochs=model.iter, total_examples=model.corpus_count)

#### Save  the model  in local  drive #############################

model.save("doc2vec.model")


#################### THIS BELOW CLASS  CONTAINS 3 FUNCTIONS .

class QABot:
    def __init__(self, data, model):
        self.df = data
        self.model = model

    def avg_feature_vector(self, sentence, model, num_features, index2word_set):

        """
        "FUNCTION TO AVERAGE  ALL  OUR FEATURE  VECTORS  IN EVERY DOCUMENT ( QUESTIONS ,IN OUR CASE )"
         sentence: Every set of data .
         model: Convert index2word_set to a set, for speed .
         index2word_set: field holds a list of the working vocabulary .
        """
        words = sentence.split()

        # Pre-initialize an empty numpy array (for speed)
        feature_vec = np.zeros((num_features,), dtype='float32')
        n_words = 0
        for word in words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec


        ###########################################################################################

    def cosine_dist(self, user_asked):

        """
            COSINE  SIMILARITY  BETWEEN  VECTORS  AND  GIVE PERCENTAGE  OF MATCH  BETWEEN  COMMON QUESTIONS  
            user_asked: users  question asked
            df: preporcessed  dataframe 
            num_features :Word vector dimensionality  
            Index2word:  field holds a list of the working vocabulary
            The model's vocabulary. Convert it to a set, for speed.
        """

        index2word_set = set(self.model.wv.index2word)
        try:
            all_ratios = []
            questn_df = pd.DataFrame({})
            for i in range(self.df.__len__()):
                s1_afv = self.avg_feature_vector(user_asked, model=self.model, num_features=100,
                                                 index2word_set=index2word_set)
                s2_afv = self.avg_feature_vector(self.df['questions'][i], model=self.model, num_features=100,
                                                 index2word_set=index2word_set)
                all_ratios.append(1 - spatial.distance.cosine(s1_afv, s2_afv))
            questn_df = pd.DataFrame({"questions": list(self.df['questions']), "answers": list(self.df['answers']),
                                      "type": list(self.df['type']),
                                      "ratios": all_ratios})
            final_ratio = questn_df.sort_values('ratios', ascending=False)

            if final_ratio.empty:

                return 'sorry  didnt  understand  your question'
            else:

                return final_ratio.head(5)
        except:
            return 'sorry  didnt  understand  your question'


            #############################################################################

    def preprocedd_user_inpt(self, user_asked):
        """
        THIS FUNCTION IS USED  TO  PREPROCESS  OUR  USER INPUT BY REMOVING  STOPWORDS  LIKE "IS" ,"AND" ,"THE" , TOKENIZE  THE  USER INPUT .
         user_asked: user intent 
        """
        a = user_asked.lower()
        stop_words = set(stopwords.words('english'))
        preprocess = []
        tknzr = TweetTokenizer()
        preprocess.append(tknzr.tokenize(a))  ### tokenize  user input
        filtered_sentence = " ".join([w for w in preprocess[0] if not w in stop_words])  ##  remove stopwords
        words = filtered_sentence
        return words


        ################################################################################


#############################################################

#################### EXECUTABLE  FUNCTION ####################
def main():

    data = pd.read_csv("preprocessed.csv",encoding="ISO-8859-1")  ### Read the preprocessed data  which is saved in local drive .
    model = Doc2Vec.load("doc2vec.model")  ####  import  doc2model model
    data['questions'].fillna("nothing", inplace=True)  ###  filling empty  values  with  'nothing' if present

    Bot = QABot(data, model)         ###Calling the class with data and the model parameters

    print('write your question below : ')  #### initial  message  shown to users
    user_ask = input()

    ############### asking  for user input from the user in below

    user_inpt = int(input('Press 1 to know the answer to your question  | Press 2 to know top 5 probabable  matching answers'))

    if (user_inpt == 1):  ######### If user input  ==1  ,  it will  show  you  the answer to the question
        userIp = Bot.preprocedd_user_inpt(user_ask)  ######## Calling the  preprocesser  function from class Bot
        predict_question = Bot.cosine_dist(user_ask).iloc[0].answers  #### Calling the cosine_dist function from class Bot
        print(predict_question)
    elif (user_inpt == 2):  ######### If user input  ==2  ,  it will  show  you  the top 5  matching  questions with probabilities .
        userIp = Bot.preprocedd_user_inpt(user_ask)  ######## Calling the  preprocesser  function from class Bot
        predict_question = Bot.cosine_dist(user_ask)[
            ['questions', 'answers', 'ratios']]  #### Calling the cosine_dist function from class Bot
        print(predict_question)

    else:
        print('please type 1 or 2 for answer')


################################################

if __name__ == '__main__':
    main()















