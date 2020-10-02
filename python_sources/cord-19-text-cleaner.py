import re
from unidecode import unidecode

class Cleaner():
    def __init__(self, is_recent=False):
        self.is_recent = is_recent
        
    re_flags = re.ASCII
    
    # https://github.com/jfilter/clean-text/blob/master/cleantext/constants.py
    URL_REGEX = re.compile(
        r"(?:^|(?<![\w\/\.]))"
        # protocol identifier
        # r"(?:(?:https?|ftp)://)"  <-- alt?
        r"(?:(?:https?:\/\/|ftp:\/\/|www\d{0,3}\.))"
        # user:pass authentication
        r"(?:\S+(?::\S*)?@)?" r"(?:"
        # IP address exclusion
        # private & local networks
        r"(?!(?:10|127)(?:\.\d{1,3}){3})"
        r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
        r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
        # IP address dotted notation octets
        # excludes loopback network 0.0.0.0
        # excludes reserved space >= 224.0.0.0
        # excludes network & broadcast addresses
        # (first & last IP address of each class)
        r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
        r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
        r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
        r"|"
        # host name
        r"(?:(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)"
        # domain name
        r"(?:\.(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)*"
        # TLD identifier
        r"(?:\.(?:[a-z\\u00a1-\\uffff]{2,}))" r")"
        # port number
        r"(?::\d{2,5})?"
        # resource path
        r"(?:\/[^\)\]\}\s]*)?",
        # r"(?:$|(?![\w?!+&\/\)]))",
        # @jfilter: I removed the line above from the regex because I don't understand what it is used for, maybe it was useful?
        # But I made sure that it does not include ), ] and } in the URL.
        flags=re_flags | re.IGNORECASE,
    )

    """
    Drop license information
    """
    LICENSE1_REGEX = re.compile(r'\(\W*which\W*was\W*(?:[^)])+reviewed\)\W+', flags=re_flags)
    LICENSE2_REGEX = re.compile(r'CC-(?:CC0-?|BY-?|SA-?|NC-?|ND-?)+\s+\d?\s?\.?\d?\s+', flags=re_flags)
    LICENSE3_REGEX = re.compile(r'International license (:?It )is made available under a\s+', flags=re_flags)
    LICENSE4_REGEX = re.compile(r'author/funder\s*|(:?URL )?doi: \w+ preprint', flags=re_flags)
    LICENSE5_REGEX = re.compile(r'The copyright holder for this preprint(:? is the)?\s*\.?\s*(?:URL\s*)?', flags=re_flags)
    LICENSE6_REGEX = re.compile(r'who has granted \w+ a license to display the preprint in perpetuity\.?', flags=re_flags)
    LICENSE7_REGEX = re.compile(r'BY-(?:SA-?|NC-?|ND-?)+\s+\d?\s?\.?\d?\s+', flags=re_flags)
    LICENSE8_REGEX = re.compile(r'No reuse allowed without permission\.?', flags=re_flags)
    LICENSE9_REGEX = re.compile(r'All rights reserved\.?', flags=re_flags)

    """
    Virus, disease names
    https://en.wikipedia.org/wiki/Novel_coronavirus
    """
    vn1 = r'sars\W{,2}cov\W{,2}2' # SARS-CoV-2
    vn2 = r'hcov\W{,2}(?:20)?19'  # HCoV-2019
    vn3 = r'(?:20)?19\W{,2}ncov'  # 2019-nCoV
    # vn4= r'Novel coronavirus' # a bit risky
    VIRUS_REGEX = re.compile(fr'{vn1}|{vn2}|{vn3}', flags=re.IGNORECASE|re_flags)
    DISEASE_REGEX = re.compile(r'covid\W{,2}(?:20)?19', flags=re.IGNORECASE|re_flags)
    
    """
    Warning we drop usefull information
    TODO: Explain why
    """
    LONG_REGEX = re.compile(r'[^\s]{64,}', flags=re_flags)
    # Should became recursive
    IN_BRACKETS_REGEX = re.compile(r'\[[^\[\]]+\]', flags=re_flags)
    IN_PARENTHESES_REGEX = re.compile(r'\([^()]+\)', flags=re_flags)
    
    LATEX_BEGIN = re.compile(r'\\begin', flags=re_flags | re.IGNORECASE)
    LATEX_END = re.compile(r'\\end', flags=re_flags | re.IGNORECASE)

    MULTI_SPACE = re.compile(r' +', flags=re_flags)
    
    regex_pre = [
        (URL_REGEX, ' URL '),

        (LICENSE1_REGEX, ' '),
        (LICENSE2_REGEX, ' '),
        (LICENSE3_REGEX, ' '),
        (LICENSE4_REGEX, ' '),
        (LICENSE5_REGEX, ' '),
        (LICENSE6_REGEX, ' '),
        (LICENSE7_REGEX, ' '),
        (LICENSE8_REGEX, ' '),
        (LICENSE9_REGEX, ' '),

        (LATEX_BEGIN, ' {'),
        (LATEX_END, '} '),

        (LONG_REGEX, ' '),
        (IN_BRACKETS_REGEX, ' '),
        (IN_PARENTHESES_REGEX, ' '),
        
        (MULTI_SPACE, ' ')
    ]
    
    """
    Recursively drop all what is inside {}
    """
    regex_rec = re.compile(r'\{[^\{\}]*\}', flags=re_flags)
    
    """
    Post processing
    Drop Latex commands like: \\begin, \\end...
    Add space around / and =
        This break some chemical names
        TODO: Show examples and/or improve
    """
    LATEX_CMD = re.compile(r'\\[^\s]+', flags=re_flags)
    AND_REGEX = re.compile(r'/', flags=re_flags)
    EQUAL_REGEX = re.compile(r'=', flags=re_flags)
    regex_post = [
        (LATEX_CMD, ' '),
        (AND_REGEX, ' / '),
        (EQUAL_REGEX, ' = ')
    ]
    
    def clean(self, txt, is_bib=False):
        """
        
        """
        
        # ASCII transliterations of Unicode text
        txt = unidecode(txt)
        
        # TODO: make apply regex function
        
        # Change virus, diease name to official names only if recent publication and not bibliography title
        if self.is_recent and not is_bib:
            txt = self.VIRUS_REGEX.sub(' SARS-CoV-2 ', txt)
            txt = self.DISEASE_REGEX.sub(' COVID-19 ', txt)
        
        for regex, replace_with in self.regex_pre:
            txt = regex.sub(replace_with, txt)
        
        max_iter = 20
        while max_iter>0:
            o=txt
            txt = self.regex_rec.sub(' ', txt)
            if o==txt:
                break
            max_iter -= 1
                
        for regex, replace_with in self.regex_post:
            txt = regex.sub(replace_with, txt)
        
        return txt.strip()