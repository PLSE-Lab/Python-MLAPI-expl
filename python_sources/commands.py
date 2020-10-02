# %% [code]
from cmd import Cmd
import spacy
from match import Match
import pandas as pd


class Commands(Cmd):
    intro = "COVID19-TM: Type ? to list commands"

    def init(self, recommender, spacy_model='en_core_sci_sm'):
        self.recommender = recommender
        self.nlp = spacy.load(spacy_model)
        self.k = 5
        self.lower = 1950
        self.upper = 2020
        self.only_covid = True

    def do_exit(self, inp):
        print("Exiting...")
        return True

    def do_set_k(self, k):
        self.k = int(k)

    def do_set_lower(self, lower):
        self.lower = lower

    def do_set_upper(self, upper):
        self.upper = upper

    def do_only_covid(self, inp):
        self.only_covid = True

    def do_all_papers(self, inp):
        self.only_covid = False

    def do_list_papers(self, k):
        if k == "":
            k = self.k
        fields = ['title', 'authors', 'url']
        df = self.recommender.mdata.ix[:, fields]
        print(df.head(int(k)).to_string())

    def do_similar_papers(self, paper_id):
        if paper_id != '':
            fields = ['title', 'authors', 'similarity']
            similar_papers = self.recommender.find_similar_papers(paper_id, self.k, self.lower, self.upper, self.only_covid)
            print(similar_papers.ix[:,fields].to_string())
        else:
            print("Missing paper ID (sha)")

    def do_text_query(self, text):
        entvocab = set([entity.text for entity in self.nlp(text).ents if len(entity.text.split(' ')) > 1])
        matcher = Match()
        matcher.matchinit_from_list(entvocab)
        tokenized = [tok.lemma_ for tok in self.nlp(text)]
        fields = ['title', 'authors', 'similarity']
        similar_papers = self.recommender.text_query(tokenized, self.k, self.lower, self.upper, self.only_covid)
        print(similar_papers.ix[:, fields].to_string())

    def do_task1(self, inp):
        task_description = "What is known about transmission, incubation, and environmental stability? What do we know" \
        "about natural history, transmission, and diagnostics for the virus? What have we learned about infection" \
        "prevention and control? Specifically, we want to know what the literature reports about: Range of incubation" \
        "periods for the disease in humans (and how this varies across age and health status) and how long individuals" \
        "are contagious, even after recovery. Prevalence of asymptomatic shedding and transmission (e.g., particularly" \
        "children). Seasonality of transmission. Physical science of the coronavirus (e.g., charge distribution," \
        "adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected" \
        "areas and provide information about viral shedding). Persistence and stability on a multitude of substrates" \
        "and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood). Persistence of virus on surfaces of" \
        "different materials (e,g., copper, stainless steel, plastic). Natural history of the virus and shedding of it"
        "from an infected person Implementation of diagnostics and products to improve clinical processes Disease " \
        "models, including animal models for infection, disease and transmission Tools and studies to monitor phenotypic" \
        " change and potential adaptation of the virus Immune response and immunity Effectiveness of movement control " \
        "strategies to prevent secondary transmission in health care and community settings Effectiveness of personal " \
        "protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community " \
        "settings Role of the environment in transmission."

        entvocab = set([entity.text for entity in self.nlp(task_description).ents if len(entity.text.split(' ')) > 1])
        matcher = Match()
        matcher.matchinit_from_list(entvocab)
        tokenized = [tok.lemma_ for tok in self.nlp(task_description)]
        fields = ['title', 'authors', 'similarity']
        similar_papers = self.recommender.text_query(tokenized, self.k, self.lower, self.upper, self.only_covid)
        print(similar_papers.ix[:, fields].to_string())

    def do_paper_info(self, paper_id):
        if paper_id in self.recommender.mdata.index:
            print(self.recommender.mdata.loc[paper_id])
        else:
            print("Incorrect paper id ({})".format(paper_id))

    def do_display_column_width(self, width):
        pd.set_option('display.max.colwidth', int(width))

    do_EOF = do_exit
