'''
Dated: Oct17-2018
Author: Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for text segmentation using tag search. 
Results:
Input dataset: seekingalpha.com roche transcript
https://seekingalpha.com/article/4190911-roche-holding-ltd-adr-rhhby-ceo-severin-schwan-q2-2018-results-earnings-call-transcript?part=single

Output:
Text segmentation based on tags of choice.

'''

import sys
import nltk
from nltk.corpus import PlaintextCorpusReader
from  nltk.text import ConcordanceIndex

def compute_concordance(tag_list, text):
    '''loop through the tag_list and compute concordance for each tag on the text'''
    for tag in tag_list:
        ci = ConcordanceIndex(text.tokens)
        concord_list_results = concordance(ci, tag, 800, sys.maxsize)
        #print("type(results)", type(list_results))
        i=0
        print("All occurences of tag: ", tag, '\n')
        
        for item in concord_list_results:
            list_of_sentences_in_item = nltk.sent_tokenize(item)    #returns a list of sentences in the item
            #remove 1st and last sentence in the item before printing
            final_list_of_sentences_in_item = list_of_sentences_in_item[1:-1]
            if len(final_list_of_sentences_in_item) !=0 :
                i+=1
                print("Occurence ", i, " of tag ", tag, '\n')
                print (final_list_of_sentences_in_item)
                print("\n")

def concordance(ci, word, width=75, lines=25):
    """
    Rewrite of nltk.text.ConcordanceIndex.print_concordance that returns results
    instead of printing them. 

    See:
    http://www.nltk.org/api/nltk.html#nltk.text.ConcordanceIndex.print_concordance
    """
    half_width = (width - len(word) - 2) // 2
    context = width // 4 # approx number of words of context

    results = []
    offsets = ci.offsets(word)
    if offsets:
        lines = min(lines, len(offsets))
        for i in offsets:
            if lines <= 0:
                break
            left = (' ' * half_width +
                    ' '.join(ci._tokens[i-context:i]))
            right = ' '.join(ci._tokens[i+1:i+context])
            left = left[-half_width:]
            right = right[:half_width]
            results.append('%s %s %s' % (left, ci._tokens[i], right))
            lines -= 1

    return results

def redirect_stdout_to_file():
    '''redirect_stdout_to_file'''
    # Open the file
    fileconcord = open('Oct17-2018-Roche-text-segmentation-results.txt', 'w')
    # Save old stdout stream
    tmpout = sys.stdout
    # Redirect all "print" calls to that file
    sys.stdout = fileconcord
    # Init the method
    print("Input dataset: seekingalpha.com roche transcript\n")
    print("https://seekingalpha.com/article/4190911-roche-holding-ltd-adr-rhhby-ceo-severin-schwan-q2-2018-results-earnings-call-transcript?part=single\n")
    
    print("\n\n\n################# FILTERING COMPANY RELATED INFORMATION #################\n\n\n")
    company_tag_list=['Roche', 'roche', 'Profit', 'profit', 'Growth', 'growth']
    compute_concordance(company_tag_list, text)
    
    print("\n\n\n################# FILTERING THERAPEUTIC AREA RELATED INFORMATION #################\n\n\n")

    therapeutic_tag_list=['Therapy', 'therapy', 'Ophthalmology', 'ophthalmology', 'Carcinoma', 'carcinoma', 'Renal', 'renal', 'Cancer', 'cancer']
    compute_concordance(therapeutic_tag_list, text)

    print("\n\n\n################# FILTERING PRODUCT/DRUG RELATED INFORMATION #################\n\n\n")
    product_tag_list=['Rituxan', 'rituxan', 'Avastin', 'avastin', 'Herceptin', 'herceptin', 'Tecentriq', 'tecentriq']
    compute_concordance(product_tag_list, text)

    print('\n################# End of program ################# \n')
    
    print('[[[[[[[[  Author: Mahesh Babu Mariappan  ]]]]]]]]]]') 
    print('[[[[[[[[  Dated: Oct17-2018              ]]]]]]]]]]') 
    print('[[[[[[[[  Contact: https://www.linkedin.com/in/mahesh-babu-mariappan ]]]]]]]]]]')
    # Close file
    fileconcord.close()
    # Reset stdout in case you need something else to print
    sys.stdout = tmpout

if __name__ == '__main__':
    '''text segmentation using tag search'''
    corp = PlaintextCorpusReader("../input/", "roche.txt")
    text = nltk.Text(corp.words())
    
    print("Input dataset: seekingalpha.com roche transcript\n")
    print("https://seekingalpha.com/article/4190911-roche-holding-ltd-adr-rhhby-ceo-severin-schwan-q2-2018-results-earnings-call-transcript?part=single\n")

    
    print("\n\n\n################# FILTERING COMPANY RELATED INFORMATION #################\n\n\n")
    company_tag_list=['Roche', 'roche', 'Profit', 'profit', 'Growth', 'growth']
    compute_concordance(company_tag_list, text)
    
    print("\n\n\n################# FILTERING THERAPEUTIC AREA RELATED INFORMATION #################\n\n\n")

    therapeutic_tag_list=['Therapy', 'therapy', 'Ophthalmology', 'ophthalmology', 'Carcinoma', 'carcinoma', 'Renal', 'renal', 'Cancer', 'cancer']
    compute_concordance(therapeutic_tag_list, text)

    print("\n\n\n################# FILTERING PRODUCT/DRUG RELATED INFORMATION #################\n\n\n")
    product_tag_list=['Rituxan', 'rituxan', 'Avastin', 'avastin', 'Herceptin', 'herceptin', 'Tecentriq', 'tecentriq']
    compute_concordance(product_tag_list, text)

    print('\n################# End of program ################# \n')
    
    print('[[[[[[[[  Author: Mahesh Babu Mariappan  ]]]]]]]]]]') 
    print('[[[[[[[[  Dated: Oct17-2018              ]]]]]]]]]]') 
    print('[[[[[[[[  Contact: https://www.linkedin.com/in/mahesh-babu-mariappan ]]]]]]]]]]')
    
    redirect_stdout_to_file()
    
    
    