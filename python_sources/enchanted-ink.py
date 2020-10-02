#!/usr/bin/env python
# coding: utf-8

# *... With my heart racing and limbs frozen, I surveyed the floor littered with crumpled scraps of paper. How am I ever going sort through this disarray? Just as my mind was smoothly transitioning from sudden shock to sheer hopelesness, a distant memory occured to me. Many a years ago, I learned an ink-enchanting spell. If I can just channel the spirits of those three frightening wordsmiths into the physical words on these pages, perhaps they will reveal some hint to who authored each sentence. I focus my mind, and slowly, word by word, spell out the incantation. As I am finishing the last syllable of the last spell, the letters on the pages come alive with a soft glow. I lean to look closer, only to be startled by a sudden flurry of motion and bright colorful flashes. Once the excitement settled down, I could see that the text transformed, and each word took on its own hue and magnitude. This must be the hint I was hoping for!*
# ![Glowing Book](https://ichef.bbci.co.uk/childrens-responsive-ichef-ck/400xn/amz/cbbc/book-swap-index.jpg)

# # 1. About This Kernel
# This kernel does not classify data for submission, but is focused purely on visualization. Any supplied text is formatted by mapping calculated metrics on font size, color, and tooltips to show how individual words indicate each of the three authors. This type of visualization can be used in multiple ways.
# 
# 1. ** As a tool for facilitated manual classification:** An expert can read the enhanced text to sort the data combining the calculated data with their personal knowledge to make the final decision.
# 2. **For analysis of classification approaches:** For example a look at misclassified sentences displayed using this formatting can reveal the reason for given sample to be misclassified, and inspire modifiction to the approach. Note that for this use case, the metrics mapped to the formatting parameters (color, size) do not necessarily have to be the same as presented in this kernel. 
# 3. **Just for fun:** Browse the data and explore what words are typical for which author. Or write your own sentence and see in who's style it is written!
# 
# # 2. Motivational Example
# Here is a sample sentence to illustrate the technique. As you look at this, try to hover your mouse over individual words to show additional info.

# In[18]:


from IPython.core.display import display, HTML # To show formatted HTML in IPython notebook  
HTML("""<font size=5 color='blue' title='\"Frankenstein\" Score (frequency)
MWS: 3.0 (16)
EAP: 0.0 (0)
HPL: 0.0 (0)'>Frankenstein</font>, <font size=5 color='green' title='\"Cthulhu\" Score (frequency)
HPL: 3.0 (14)
EAP: 0.0 (0)
MWS: 0.0 (0)'>Cthulhu</font>, <font size=1 color='green' title='\"and\" Score (frequency)
HPL: 1.138 (5494)
MWS: 1.055 (5506)
EAP: 0.807 (5184)'>and</font> <font size=5 color='red' title='\"Dupin\" Score (frequency)
EAP: 3.0 (52)
HPL: 0.0 (0)
MWS: 0.0 (0)'>Dupin</font> <font size=2 color='green' title='\"walk\" Score (frequency)
HPL: 1.673 (22)
MWS: 0.984 (14)
EAP: 0.343 (6)'>walk</font> <font size=1 color='red' title='\"into\" Score (frequency)
EAP: 1.23 (417)
HPL: 1.004 (256)
MWS: 0.765 (211)'>into</font> <font size=1 color='red' title='\"a\" Score (frequency)
EAP: 1.137 (4256)
HPL: 1.065 (2996)
MWS: 0.798 (2428)'>a</font> <font size=2 color='red' title='\"bar\" Score (frequency)
EAP: 1.669 (5)
HPL: 1.331 (3)
MWS: 0.0 (0)'>bar</font>.""")


# Each author is assigned a color: blue for M. W. Shelley, green for H. P. Lovecraft and red for E. A. Poe. A cursory look at the formatted text should give a reader an idea about who wrote the sentence, albeit in this example it would be rather challenging. Let's look at the first few  words in the above sentence.
# - **Frankenstein** is blue and large, meaning it is a very "MWS" word. When hovering mouse over it, the tooltip shows that MWS used Frankenstein 16 times while the other authors did not use it at all. This results in a max author score of 3.0, also shown in the tooltip.
# - **Cthulhu** is green and large, meaning it is a very "HPL" word. Again, it has author score of 3.0, since it is exclusive to HPL.
# - **and** is written in small green font. Green means that it is most likely for HPL, nonetheless its small size shows that other authors use it almost as much. Look at the tooltip reveals that HPL's winning author score is only 1.138.
# - ...

# # 3. Word Metrics Calculation
# 
# As hinted above, the size and color of individual words is determined by the author score metric. It is be calculated sing the following simple calculation.  
# 
# First, we define the propensity of word $w$ in the corpus of author $a$.
# $$ p_w^a = \frac{f_w^a}{N^a} $$
# Here, $N^a$ is total number of words in author $a$'s data sample and $f_w^a$ is number of times word $w$ appeared therein.
# 
# Final score for word $w$ and author $a$ is then calculated as propensity for given author divided by the average of all the authors' propensities. 
# $$ Score_w^a = \frac{p_w^a}{\frac{1}{3}\left( p_w^{EAP}+p_w^{MWS}+p_w^{HPL}\right)}  $$
# 
# With the three author scores ready, we can apply formating to word $w$.
# - The author with the highest score is mapped to font color (*Who is the most likely author of this word?*)
# - The highest score is mapped to font size (*How strongly does the word indicate the author selected above?*)
# - All three author scores along with word frequencies are displayed in the tooltip   

# # 4. Implementation
# 
# ## 4.1 - Read in the data
# 
# Create three dataframes ...
# *  **Training data** - Randomly selected 90% of the labeled input data.
# * **Validation data** - Randomly selected 10% of the labeled input data.
# * **Test data** - All of the unlabeled input data.

# In[19]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk # the natural langauage toolkit, open-source NLP
from IPython.core.display import display, HTML # To show formatted HTML in IPython notebook  
from sklearn.model_selection import train_test_split # To create a validation dataset

# read the full labeled dataset
texts = pd.read_csv("../input/train.csv")
# Create a 10% validation and 90% training datasets
texts_train, texts_valid = train_test_split(texts, test_size=0.1, random_state=5675)

# read the unlabeled dataset
texts_test = pd.read_csv("../input/test.csv")


# ## 4.2 - Calculate Word Frequencies by Author
# 
# This process feeds the training sentences into nltk's ConditionalFreqDist object, so that we can easily find out how many times was a word used by each of the three authors. This code is taken directly from the [sample kernel for this challenge](https://www.kaggle.com/rtatman/beginner-s-tutorial-python). Thank you!

# In[20]:


# split the data by author
byAuthor = texts_train.groupby("author")

# word frequency by author
wordFreqByAuthor = nltk.probability.ConditionalFreqDist()

# for each author...
for name, group in byAuthor:
    # get all of the sentences they wrote and collapse them into a
    # single long string
    sentences = group['text'].str.cat(sep = ' ')
    # convert everything to lower case (so "The" and "the" get counted as 
    # the same word rather than two different words)
    sentences = sentences.lower()
    # split the text into individual tokens    
    tokens = nltk.tokenize.word_tokenize(sentences)
    # calculate the frequency of each token
    frequency = nltk.FreqDist(tokens)
    # add the frequencies for each author to our dictionary
    wordFreqByAuthor[name] = (frequency)
    
# now we have a dictionary where each entry is the frequency distrobution
# of words for a specific author.


# ## 4.3 Word Score Calculation Example
# 
# This section is not part of the final implementation, but illustates on examples how author scores are calculated. The calculation can be done easily thanks to the nltk's ConditionalFreqDist class. Previously, we have created its instance in the object wordFreqByAuthor and loaded it with our training data. Word propensity, as we defined it above, can be directly retrieved using the method wordFreqByAuthor[*author*].freq(*word*), so the score calculation is straightforward. We will use the following function to show the calculation on a few example words.

# In[21]:


def illustrateScoreCalculation(word):
    """Calculate word scores for all three authors, print results in console."""
    authors = ['EAP','HPL','MWS']
    # Get word frequency for all three authors
    frequencies = {a:wordFreqByAuthor[a][word] for a in authors}
    # Get propensity for all three authors
    propensities = {a:wordFreqByAuthor[a].freq(word) for a in authors}
    # Calculate average propensity 
    avgPropensity =sum(propensities.values())/3
    # Calculate final scores by dividing author's propensity by average propensity
    scores = {a:propensities[a]/avgPropensity for a in authors}
    # Show what we have learned
    for a in authors:
        print("{0} used the word \"{1}\" {2}x, final score is {3:.6f}/{4:.6f} = {5:.2f}".format(
                a,word,frequencies[a],propensities[a],avgPropensity,scores[a]))


# In[22]:


### Example word 1 ###
illustrateScoreCalculation("drawer")


# Obviously E. A. Poe uses the word "drawer" much more than the other two authors. This reflects in his score being 2.76, close to the max value of 3.0. 

# In[23]:


### Example word 2 ###
illustrateScoreCalculation("card")


# The word "card" was only used by one author in the training set - E. A. Poe. This means his score is the maximum value of 3.00.

# In[24]:


### Example word 3 ###
illustrateScoreCalculation("and")


# H. P. Lovecraft uses the word "and" most frequently in his writing. Nonetheless, the other two authors use it almost as much, therefore the winning score is only 1.14, close to the minimum of 1.0.

# ## 4.4 - Define EnchantedInkPrinter Class
# 
# This class combines two main functions: calculation of author scores as described in section 3 above, and generation of formatted html code that maps the calculated author scores to printed font properties.
# 

# In[25]:


class EnchantedInkPrinter:
    # Colors to use for the three authors
    colors = ["red","green","blue"]
    # What color to print words that did not appear in the training data
    neutralColor = "black" 
    # Text sizes to be mapped to the author score metric
    textSizeSteps = [5,4,3,2,1]
    # Add a constant to all of the textSizeSteps to make output larger
    textSizeOffset = 0
    # Format author's code (EAP, MWS, HPL) if it appears in text
    formatClassNames = True
    # Add color and size legend to printed html text
    showLegend = True
    # Experimental: Calculate the most likely author of the whole
    # sentence based on summary of individual word scores
    showSentenceSummaryScore = False
    """Provides methods to print an input string or dataset as a formatted 
    HTML output that shows the most likely author for each word based on
    provided ConditionalFreqDist object."""
    
    def __init__(self, classFreq):
        """parameter classFreq: nltk.probability.ConditionalFreqDist containing word 
        frequencies for the analyzed authors"""
        self.classFreq = classFreq
        # Number of authors
        classCount = len(classFreq.conditions())
        # Only can have as many authors as elements in self.colors (this can be expanded easily)
        if classCount > len(self.colors):
            raise ValueError('Maximum number of classes exceeded: {0} > {1}'.format(classCount,len(self.colors)))
        elif classCount < 2:
            raise ValueError('There must be at least two classes')
        # Assign a color to each class
        self.palette = {a:b for a,b in zip(classFreq.conditions(),self.colors)}
        self.palette[''] = self.neutralColor
        # Calculate text size cutoffs by spreading number of self.textSizes
        # over the whole max author score spectrum of 1.0 - 3.0 (# of authors)
        self.maxScore= float(classCount)
        scoreStep = (classCount-1.0)/(len(self.textSizeSteps)-1)
        self.textSizes = []
        for i in range(len(self.textSizeSteps)-1):
            self.textSizes.append( (self.maxScore-scoreStep*i,self.textSizeSteps[i]) )
        self.textSizes.append((0.0,self.textSizeSteps[-1]))
    
    def __calcWordScore(self,word):
        """Calculate author metrics for one word
        parameter word: string with a single word
        returns: all word metrics in this structure:
                 {'word':<word>,
                  'classScores':[ {'class':<author1 name>, 
                                   'score':<author1 score>, 
                                   'frequency':<author1 frequency> }, 
                                  {'class':<author2 name>, 
                                   'score':<author2 score>, 
                                   'frequency':<author2 frequency>}
                                  {'class':<author3 name>, 
                                   'score':<author3 score>, 
                                   'frequency':<author3 frequency>}]}
                Note: the 'classScores' list is sorted by descending author score, so the most
                      probable author is always on the first position.
        """
        scoreList = None
        # Get average propensity of given word among all authors
        overallWordFreq = self.__getOverallWordFreq(word)
        # For author codes (EAP, MWS, HPL), create synthetic score
        if self.formatClassNames and word in self.classFreq.conditions():
            scoreList = [{'class':word,'score':self.textSizes[0][0]+0.1}]
        # If the file was not used in the training data, return empty score
        elif overallWordFreq==0:
            scoreList = [{'class':'','score':0.0}]
        # Otherwise calculate author scores
        else:
            scoreList = []
            for a in self.classFreq.conditions():
                # Score calculation
                score = self.classFreq[a].freq(word.lower())/overallWordFreq
                # Frequency calculation
                frequency = self.classFreq[a][word.lower()]
                # For terms that were used only once in the whole training data,
                # adjust score by subtracting 0.01
                if score == self.maxScore and frequency == 1:
                    score -= 0.01
                scoreList.append({'class':a,'score':score,'frequency':frequency})
            # Sort by descending score
            scoreList.sort(key=lambda x:x['score'],reverse=True)
        return {'word':word,'classScores':scoreList}
    
    def __getOverallWordFreq(self,word):
        """Calculate average propensity of given word among all authors""" 
        sumScore = 0.0
        for a in self.classFreq.conditions():
            sumScore += self.classFreq[a].freq(word.lower())
        return sumScore / len(self.classFreq.conditions())

    def __calcSentenceScore(self,text):
        """Take sentence provided in parameter text, tokenize it to individual 
        words, get metrics for each of the words (in structure as described 
        in __calcWordScore) and then return an array of all word metrics. 
        """ 
        scores = []
        for word in nltk.tokenize.word_tokenize(text):
            score = self.__calcWordScore(word)
            # append the returned word score dictionary to the scores array
            scores.append(score)
        # Experimental: Summarize scores for all words and add an extra word 
        # to the return list with the summary information
        if self.showSentenceSummaryScore:
            summaryScore =self.__aggregateSentenceScore(scores)
            scores.append({'word':'['+summaryScore[0]['class']+']','classScores':summaryScore})
        return scores
    
    def __aggregateSentenceScore(self,scores):
        """Experimental: Based on list of scores for all words in sentence,
        calculate overall sentence score that should indicate the most likely 
        author for the whole sentence. Return a synthetic word score with the
        summary information that can be added to the sentence when printing."""
        # In the summary score, score represents sum of scores 
        # of all words in the sentence for given author         
        scoreTotals = {a:0.0 for a in self.classFreq.conditions()}
        # In the summary score, frequency represents the number 
        # of words that given author 'has won' in the sentence
        freqTotals = {a:0 for a in self.classFreq.conditions()}
        for score in scores:
            if score["classScores"][0]['class'] != '':                 
                freqTotals[score["classScores"][0]['class']] += 1
                for a in score["classScores"]:
                    scoreTotals[a['class']] += a['score']
        scoreList = [{'class':c,'score':scoreTotals[c],'frequency':freqTotals[c]} for c in scoreTotals.keys()]
        scoreList.sort(key=lambda x:x['score'],reverse=True)
        return scoreList

    def __getSentenceHtml(self,wordScores):
        """Based on the input list of word scores in wordScores, create formatted html
        code that maps given scores to font color and size. The element of wordScores
        are word score dicts as described in __calcWordScore """
        line = ""
        # This loops over the words in the sentence
        for wordScore in wordScores:
            word = wordScore['word']
            scores = wordScore['classScores']
            bestClass = scores[0]['class']
            bestScore = scores[0]['score']
            # Do not format interpunction, just add it to the ouptut
            if word in (",",";",".","'",'"'):
                line += word
            else:
                # Find the text size for this word
                textSize = 0
                for scoreCutoff,size in self.textSizes:
                    if bestScore >= scoreCutoff:
                        textSize = size
                        break
                # Construct the tooltip text with all information
                tooltipText = ""
                if len(scores) > 1:
                    tooltipText = "title='\"" + word + "\" Score (frequency)"
                    for classScore in scores:
                        tooltipText += "\n{0}: {1} ({2})".format(classScore['class'],                                                                 round(classScore['score'],3),                                                                 classScore['frequency'])
                    tooltipText += "'"
                # Create the final html code for this word including the proper <font> tags
                line += " <font size={0} color='{1}' {3}>{2}</font>"                    .format(textSize + self.textSizeOffset,self.palette[bestClass],word,tooltipText)
        return line
    
    def __getCombinedLegendHtml(self):
        """Create HTML code with table with color and size legend next to each other"""
        htmlCode = '<table><tr><td>Color Legend (Author with highest score)</td><td>Size Legend</td></tr><tr><td>'
        htmlCode += self.__getColorLegendHtml() + '</td><td>' 
        htmlCode += self.__getSizeLegendHtml() + '</td></tr></table>'
        return htmlCode
        
    def __getColorLegendHtml(self):
        """Create HTML code for color legend"""
        # Create synthetic word score list, then use __getSentenceHtml to generate the actual HTML code
        wordScoreList = [{'word':w+'<br>','classScores':[{'class':w,'score':self.textSizes[0][1]}]} for w in self.classFreq.conditions()]
        return self.__getSentenceHtml(wordScoreList)
        
    def __getSizeLegendHtml(self):
        """Create HTML code for size legend"""
        # Create synthetic word score list, then use __getSentenceHtml to generate the actual HTML code
        sizeLegendElements = []
        lastScore = None
        for score,size in self.textSizes:
            if lastScore is None:
                sizeLegendElements.insert(0, {'word':'score {0}'.format(score),                                              'classScores':[{'class':'','score':score}]})
            else:
                sizeLegendElements.insert(0, {'word':'score between {0} and {1}<br>'.format(score,lastScore),                                              'classScores':[{'class':'','score':score}]})
            lastScore = score
        return self.__getSentenceHtml(sizeLegendElements)
    
    def printString(self,text):
        """Take a word or a sentence in the parameter text, calculate author scores,
        and print it formatted according to the author scores """
        scoredWords = self.__calcSentenceScore(text)
        htmlCode =self.__getSentenceHtml(scoredWords)
        if self.showLegend:
            htmlCode += self.__getCombinedLegendHtml()
        display(HTML(htmlCode))
        
    def printDataFrame(self,data):
        """Take a data frame in the parameter data. Print the whole data frame,
        formatting all text columns using calculated author scores"""
        htmlCode = ""
        if self.showLegend:
            htmlCode = self.__getCombinedLegendHtml()
        htmlCode += '<table><tr>'
        for colName in data.columns:
            htmlCode += '<td>' + colName + '</td>'
        htmlCode += '</tr>'
        for index,row in data.iterrows():
            htmlCode += "<tr>"
            for value in row:
                if isinstance(value, str):
                    htmlCode += '<td>' + self.__getSentenceHtml(self.__calcSentenceScore(value)) + '</td>'
                else:
                    htmlCode += str(value)
            htmlCode += "</tr>"
        display(HTML(htmlCode))


# ## 4.5 Try out the EnchantedInkPrinter Class
# 
# The following code snippet shows the usage of the EnchantedInkPrinter class. 

# In[26]:


# Initialize an instance of EnchantedInkPrinter
# The constructor parameter is wordFreqByAuthor that we loaded with training data earlier
printer = EnchantedInkPrinter(wordFreqByAuthor)

# The class exposes two public methods
#   - printString(<str>) To print a sentence or word in a string.
#   - printDataFrame(<DataFrame>) To print a whole dataframe, formatting all character columns

# Let's try out the former function on the motivational example from section 2
printer.printString("Frankenstein, Cthulhu and Dupin walk into a bar.")


# # 5. Challenge Data Visualization
# 
# With all the explanations out of the way, we can now look at the actual data.
# 
# ## 5.1 Validation Data Visualization
# In section 4.1, we have taken random 10% of the challenge training data to use as a validation dataset. That means we can now use it to show the visualization on a fresh set of data (that was not used for training) and see what the actual labels are. You can browse through the data and see how well the formatting indicates the author of each of the sentences.

# In[27]:


# Print the first 20 rows of the validation dataset
# Modify the subset cutoffs as desired to show other data rows! 
printer.printDataFrame(texts_valid[0:30])


# ## 5.2 Test Data Visualization
# Here are some rows from the challenge test dataset with no labels attached. Can you guess what the labels should be?

# In[28]:


# Print the first 20 rows of the validation dataset
# Modify the subset cutoffs as desired to show other data rows! 
printer.printDataFrame(texts_test[0:30])

