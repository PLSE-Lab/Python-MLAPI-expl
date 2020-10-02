import nltk
import pandas as pd 
import  numpy as np
from clean import *
from collections import Counter

class BagWords:
    
    @staticmethod
    def get_key_from_dict_by_value(dictionary, value):
        """
        Returns a key from a dictionary by value
        
        input:
        dictionary: dict
        
        value: str/ int 
            
        return:
        key: str/ int
        
        child func:
            BagWords.word_guess
            BagWords.sentences_guess
        """
        
        for key in dictionary.keys():    
            if value in dictionary[key]:
                return key
    
    
    @staticmethod
    def word_guess(bag_words, i, stem_a):
        """
        Analyzes the word and searches for the appropriate key in the dictionary 
        (searches for the appropriate word)
        
        input:
        bag_words: dict
            {"categty":["stem(word)","stem(word)", "stem(word)"],
            "critical":["critic", "intens", "sever"]}
            
        i: str
            word/words
        
        stem_a: nltk.stem
            For example: stem_a = nltk.stem.SnowballStemmer('english')
            
        return:
        key: ls
            the guess\Guesses of category 
        
        mam func:
            BagWords.get_key_from_dict_by_value
            
        child func :

        """
        guess = []
        for x in i:
            x = stem_a.stem(x)
            category = BagWords.get_key_from_dict_by_value(bag_words, x)
            if category != None:
                guess.append(category)
        return guess
        
    
#     @staticmethod
#     def expressions_guess( i, stem_a, expressions_bag):
#         """
#         Analyzes the word and searches for the appropriate key in the dictionary 
#         (searches for the appropriate -expressions-)
        
#         input:
#         i: str
#             word/words
        
#         stem_a: nltk.stem
#             For example: stem_a = nltk.stem.SnowballStemmer('english')
            
#          expressions_bag: dict
#                 {"good":['not', 'hospit','in', 'progress'],
#                 "categty":["stem(word)","stem(word)","stem(word)","stem(word)"]}

#         return:
#         key: ls
#             the guess\Guesses of category 
            
            
            
#         mam func:
#             BagWords.get_key_from_dict_by_value
            
#         child func:
#         """
#         guess = []
#         ls_i = [stem_a.stem(x) for x in i]
#         guess.append(BagWords.get_key_from_dict_by_value(expressions_bag, ls_i))
#         return guess
    
    
    @staticmethod
    def multi_expressions_guess(i, stem_a, expressions_bag, num_comparison):
        """
        Cataloging Expressions in Free Text by Expression Dictionary

        input:
        i:str
            text you want to categorize

        stem_a: nltk.stem
                For example: stem_a = nltk.stem.SnowballStemmer('english')

         expressions_bag: dict
                {"good":['not', 'hospit','in', 'progress'],
                "categty":["stem(word)","stem(word)","stem(word)","stem(word)"]}

         num_comparison:int 
             The number of words to identify from the list to classify the text as the specific phrase

        return:
        guess:list
            The guess of the phrase (can be several) according to the dictionary
                """
        op = []
        for x in i:
            x = stem_a.stem(x)
            for k, v in expressions_bag.items():
                for l in v:
                    if l == x:
                        op.append(k)

        test_dict = dict(Counter(op))

        guess = []
        for k, v in test_dict.items():
            if v >= num_comparison:
                guess.append(k)

        return guess
    
    @staticmethod
    def guess_categories_supervision(i, guess, guess_w, guess_e, indx):
        """
        child func:
            BagWords.guess_categories
        """
        error_index = []
        # The total number of objects in a sentence that do not consist of letters
        word_in_i = [x.isalpha() for x in i]
        num_objects_in_i = len([0 for x in word_in_i if  x == False])
        
        # Difference between the number of words in a sentence from the expected number of guesses
        s = len(i) - len(guess_e)*2 - len(guess_w) - num_objects_in_i

        if s > 0 :
            print(indx)
            print("expected - found = "+ str(s))
            print("words num ", str(len(i) - num_objects_in_i))
            print(i)
            print(guess)
            print("expressions " + str(len(guess_e)*2), "words "+ str(len(guess_w)))
            print()
            error_index.append(indx)
        return error_index
    
    
    @staticmethod 
    def guess_categories(df, input_col, output_col, indexs, stem_a, bag_words, sentences_bag,
                             num_comparison, supervision):
            """
            input:
            df: pd.df

            input_col: str
                name of column A

            output_col: str
                name of column B
                A ==B or A!=B

            indexs: pd.index
                index contain data col.notnull()
                you can drop some contain data indexs too 

            stem_a: nltk.stem
                For example: stem_a = nltk.stem.SnowballStemmer('english')

            bag_words: dict
                {"categty":["stem(word)","stem(word)", "stem(word)"],
                "critical":["critic", "intens", "sever"]}

            sentences_bag: dict
                {"good":[['not', 'hospit'],['in', 'progress']],
                "categty":[["stem(word)","stem(word)"],["stem(word)","stem(word)"]]}

            num_comparison:int

            supervision:bool 


            return:
                error_index:ls
                    list of index that there is not guess for them 
                    
            mam func:
                BagWords.word_guess
                BagWords.multi_expressions_guess
                Clean.organize_ls_to_str
            
            child func:
                BagWords.basic_nlp
                    """
            
            for indx in indexs:
                i = df.loc[indx, input_col]
                
                # clean str and make a list 
                i = Clean.clean_str_replace(i, [",", ".", ";", ":", "-", "‚", "+"], " ")
                i = i.split(" ")
                i = Clean.remove_from_ls(i ,"")
                
                # guess word & expressions
                guess_w = BagWords.word_guess(bag_words, i, stem_a)
                guess_e = BagWords.multi_expressions_guess(i, stem_a, sentences_bag, num_comparison)
                
                # United results of the guesswork
                guess = guess_e + guess_w
                guess = Clean.organize_ls_to_str(guess)

                if supervision == False:
                    df.loc[indx, output_col] = guess

                elif supervision == True:
                    BagWords.guess_categories_supervision(i, guess, guess_w, guess_e, indx)
#      #             -Supervision while building the word bags-

#                     ab = [x.isalpha() for x in i]

#                     ab_num = len([0 for x in ab if  x == False])

#                     s = len(i) - len(guess_s)*2 - len(guess_w) - ab_num

#                     if s > 0 :
#                         print(indx, s)
#                         print(i, len(i), ab_num)
#                         print(guess)
#                         print(len(guess_s)*2, len(guess_w))
#                         print()
#                         error_index.append(indx)
#                     return error_index
                
                


    @staticmethod 
    def basic_nlp(dfs, input_col, output_col, bag_words, expressions_bag , num_comparison, supervision):
        """
        Uses word_bag and  expressions_bag to parse column A information and returns a result to column B

        input:
        df: pd.df

        input_col: str
            name of column A

        output_col: str
            name of column B
            A == B or A!=B

        bag_words: dict
            {"categty":["stem(word)","stem(word)", "stem(word)"],
            "critical":["critic", "intens", "sever"]}

        expressions_bag: dict
            {"good":[['not', 'hospit'],['in', 'progress']],
            "categty":[["stem(word)","stem(word)"],["stem(word)","stem(word)"]]}

        num_comparison:int

        supervision:bool 
        dfs:list  

        return:
        error_index:ls
            list of index that there is not guess for them 

        mam func:
            BagWords.multi_guess_category
        """
        for df in dfs:
            # replace "" to np.nan 
            df[input_col] = df[input_col].apply(lambda x: np.nan if x== "" else x)
            
            # create output col , ps, index.notnull() 
            df[output_col] = np.nan
            ps = nltk.stem.SnowballStemmer('english')
            indexs = df.index[df[input_col].notnull()]

            # guess categories for i in col 
            error_index = BagWords.guess_categories(df , input_col, output_col, indexs, ps , 
                                 bag_words, expressions_bag, num_comparison, supervision)
            # replace "" to np.nan 
            df[output_col] = df[output_col].apply(lambda x: np.nan if x== "" else x)

            print(error_index)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     @staticmethod
#     def guess_category(df, input_col, output_col, indexs, stem_a, bag_words, sentences_bag):
#         """
#         Takes a column A trying to guess a case category
#         according to 2 dictionaries data Bag word and Bag phrase  (sentences)
#         and write the category in column B
        
#         input:
#         df: pd.df
        
#         input_col: str
#             name of column A
        
#         output_col: str
#             name of column B
#             A ==B or A!=B
        
#         indexs: pd.index
#             index contain data col.notnull()
#             you can drop some contain data indexs too 
        
#         stem_a: nltk.stem
#             For example: stem_a = nltk.stem.SnowballStemmer('english')
        
#         bag_words: dict
#             {"categty":["stem(word)","stem(word)", "stem(word)"],
#             "critical":["critic", "intens", "sever"]}
            
#         sentences_bag: dict
#             {"good":[['not', 'hospit'],['in', 'progress']],
#             "categty":[["stem(word)","stem(word)"],["stem(word)","stem(word)"]]}
            
#         return:
#             no_guess:ls
#                 list of index that there is not guess for them 
#             multi_guess: ls
#                 list of index that there are many guess for them
                
#         mam func:
#             BagWords.clean_str
#             BagWords.word_guess
#             BagWords.sentences_guess
#         """
#         multi_guess = []
#         no_guess = []

#         for indx in indexs:
#             i = df.loc[indx, input_col]
#             i = Clean.clean_str_replace(i, [",", ".", ";", ":", "-", "‚", "+"], " ")
#             i = i.split(" ")
#             guess = BagWords.word_guess(bag_words, i, stem_a)

#             if len(guess) == 1:
#                 df.loc[indx, output_col] = guess[0]

#             elif len(guess) == 0: 
#                 guess_s = BagWords.sentences_guess(sentences_bag, i, stem_a)
#                 df.loc[indx,output_col] = guess_s[0]
#                 if  len(guess_s) == 0:
#                     no_guess.append(indx)

#             elif len(guess) > 1:
#                 multi_guess.append(indx)
#         return no_guess, multi_guess
    
    
    @staticmethod
    def v():
        print(19)
