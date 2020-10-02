'''
Dated: Nov09-2018
Auhor: Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for text processing Named Entity Recognizer NER
'''
import sys
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.text import ConcordanceIndex
from nltk.text import Text


def OCR():
    # import Image
    # from tesseract import image_to_string
    try:
        from PIL import Image
    except ImportError:
        import Image
    import pytesseract
    """ print(image_to_string(Image.open('../input/dataset-scanned-documents-ocr/scannedpaper1-png.png')))
    print(image_to_string(Image.open('test-english.jpg'), lang='eng')) """
    print(pytesseract.image_to_string(Image.open('../input/dataset-scanned-documents-ocr/scannedpaper1-png.png')))

def using_spacy():
    import spacy
    nlp = spacy.load('en')

    # doc = nlp("Next week I'll be in Madrid.")
    """doc = nlp('''Andrew Yan-Tak Ng is a Chinese American computer scientist.
    He is the former chief scientist at Baidu, where he led the company's
    Artificial Intelligence Group. He is an adjunct professor (formerly 
    associate professor) at Stanford University. Ng is also the co-founder
    and chairman at Coursera, an online education platform. Andrew was born
    in the UK in 1976. His parents were both from Hong Kong.''')"""
    doc = nlp('''Wipro won Gold Award for ‘Integrated Security Assurance Service (iSAS)’ under the ‘Vulnerability Assessment, Remediation and Management’ category of the 11th Annual 2015 Info Security PG’s Global Excellence Awards.[90]
Wipro won 7 awards, including Best Managed IT Services and Best System Integrator in the CIO Choice Awards 2015, India[91]
In 2014, Wipro was ranked 52nd among India's most trusted brands according to the Brand Trust Report, a study conducted by Trust Research Advisory.[92]
Wipro was ranked 2nd in the Newsweek 2012 Global 500 Green companies.[93]
Wipro received the 'NASSCOM Corporate Award for Excellence in Diversity and Inclusion, 2012', in the category 'Most Effective Implementation of Practices & Technology for Persons with Disabilities'.[94]
In 2012, it was awarded the highest rating of Stakeholder Value and Corporate Rating 1 (SVG 1) by ICRA Limited.[95]
It received National award for excellence in Corporate Governance from the Institute of Company Secretaries of India during the year 2004.''')    
    for ent in doc.ents:
        print(ent.text, ent.label_)
     
    # Output: 
    # Next week DATE
    # Madrid GPE
    

def named_entity_recognizer_progdef_data():
    # this is NER on program-defined text
    doc = '''Andrew Yan-Tak Ng is a Chinese American computer scientist.
    He is the former chief scientist at Baidu, where he led the company's
    Artificial Intelligence Group. He is an adjunct professor (formerly 
    associate professor) at Stanford University. Ng is also the co-founder
    and chairman at Coursera, an online education platform. Andrew was born
    in the UK in 1976. His parents were both from Hong Kong.'''
    print("type(doc):", type(doc))
    
    # tokenize doc
    tokenized_doc = nltk.word_tokenize(doc)
     
    # tag sentences and use nltk's Named Entity Chunker
    tagged_sentences = nltk.pos_tag(tokenized_doc)
    ne_chunked_sents = nltk.ne_chunk(tagged_sentences)
     
    # extract all named entities
    named_entities = []
    for tagged_tree in ne_chunked_sents:
        if hasattr(tagged_tree, 'label'):
            entity_name = ' '.join(c[0] for c in tagged_tree.leaves()) #
            entity_type = tagged_tree.label() # get NE category
            named_entities.append((entity_name, entity_type))
    #print(named_entities)
    
    person_related = []
    organization_related = []
    location_related = []
    for item in named_entities:
        #print(item, item[0], item[1])
        if item[1] == 'PERSON':
            person_related.append(item[0])
        elif item[1] == 'ORGANIZATION':
            organization_related.append(item[0])
        elif item[1] == 'GPE':
            location_related.append(item[0])
    print("person_related: ", person_related)
    print("organization_related: ", organization_related)
    print("location_related: ", location_related)
    
def info_extraction_using_concordance():
    # concordance on file
    corp = PlaintextCorpusReader("../input/dataset-text-formal-documents-for-ner/", "NER-named-entity-recognition.txt")
    text = nltk.Text(corp.words())
    
    # match = text.concordance('James')
    
    # concordance on string
    doc = '''Andrew Yan-Tak Ng is a Chinese American computer scientist.
    He is the former chief scientist at Baidu, where he led the company's
    Artificial Intelligence Group. He is an adjunct professor (formerly 
    associate professor) at Stanford University. Ng is also the co-founder
    and chairman at Coursera, an online education platform. Andrew was born
    in the UK in 1976. His parents were both from Hong Kong.'''    
    tokens = nltk.word_tokenize(doc)
    myTextdoc = Text(tokens)
    match = myTextdoc.concordance('Ng')
    
    
    
    #concord = text.concordance('Herceptin', 300, sys.maxsize)
    
    """ # write Herceptin concordance to file
    # Open the file
    fileconcord = open('containing-tag-James.txt', 'w')
    # Save old stdout stream
    tmpout = sys.stdout
    # Redirect all "print" calls to that file
    sys.stdout = fileconcord
    # Init the method
    text.concordance("James", 250, sys.maxsize)
    # Close file
    fileconcord.close()
    # Reset stdout in case you need something else to print
    sys.stdout = tmpout"""
            
def wipr_data_processor():
    '''use this method to process wipr text data'''
    print("hi, I am the wipr data processor")

def named_entity_recognizer_textfile_data(line):
    # this is NER on text file data
    
    tokenized_line = nltk.word_tokenize(line)
     
    # tag sentences and use nltk's Named Entity Chunker
    tagged_sentences = nltk.pos_tag(tokenized_line)
    ne_chunked_sents = nltk.ne_chunk(tagged_sentences)
     
    # extract all named entities
    named_entities = []
    for tagged_tree in ne_chunked_sents:
        if hasattr(tagged_tree, 'label'):
            entity_name = ' '.join(c[0] for c in tagged_tree.leaves()) #
            entity_type = tagged_tree.label() # get NE category
            named_entities.append((entity_name, entity_type))
    print(named_entities)

if __name__ == '__main__':
    '''Source code for text processing Named Entity Recognizer NER'''
    #read in the data from text file
    """corp = PlaintextCorpusReader("../input/", "roche.txt")
    text = nltk.Text(corp.words())
    print("type(corp):", type(corp))
    print("type(text):", type(text))"""
    #lines = [line.rstrip('\n') for line in open('../input/dataset-text-formal-documents-for-ner/NER-named-entity-recognition.txt')]
    lines1 = [line for line in open('../input/dataset-text-formal-documents-for-ner/NER-named-entity-recognition.txt')]
    lines2 = [line for line in open('../input/dataset-roche-call-transcript-text/roche.txt')]
    lines3 = ["Wipro Limited is an India-based information technology, consulting and business process services company headquartered in Bengaluru, India",
                "In 2013, Wipro demerged its non-IT businesses into separate companies.", 
                "In 2017, Wipro Limited won a five-year IT infrastructure and applications managed services engagement with Grameenphone (GP), a leading telecom operator in Bangladesh and announced it would set up a new delivery centre there.",
                "Abidali Neemuchwala was appointed as Wipro's CEO after T. K. stepped down in early 2016.Neemuchwala, who had been group president and COO from April 2015, was appointed CEO with effect from 1 February 2016.",
                "In March 2017, Wipro was recognized as one of the world’s most ethical companies by US-based Ethisphere Institute for the sixth consecutive year."] 

    #named_entity_recognizer_progdef_data()
    '''for line in lines3[:]: 
        #print(line)
        named_entity_recognizer_textfile_data(line)'''
    
    #info_extraction_using_concordance()
    # OCR()
    using_spacy()
        