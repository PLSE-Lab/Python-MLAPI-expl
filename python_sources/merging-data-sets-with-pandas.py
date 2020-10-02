import pandas as pd

PAPERS_FILE = "/Users/mikeghen/Downloads/Papers.tsv"
AFFILIATIONS_FILE = "/Users/mikeghen/Downloads/Affiliations.tsv"
PAPER_AUTHOR_AFFILIATIONS_FILE = "/Users/mikeghen/Downloads/PapersAuthorsAffiliations.tsv"

def main():
    dataframe = denormalized_data()
    dataframe = dataframe.groupby(['conference_abbrv', 'year', 'affiliation_name'], as_index=False)
    dataframe = dataframe['paper_id'].count()
    dataframe = dataframe.rename(columns={'paper_id': 'paper_count'})
    dataframe = dataframe.sort_values(['conference_abbrv', 'year', 'paper_count'], ascending=False)
    dataframe.to_csv('results.csv')


def denormalized_data():
    affiliations_df = pd.read_csv(AFFILIATIONS_FILE, delimiter='\t', header=None,
                                  names=["affiliation_id", "affiliation_name"])

    papers_df = pd.read_csv(PAPERS_FILE, delimiter='\t', header=None,
                                  names=["paper_id", "paper_title", "year",
                                         "conference_id", "conference_abbrv"])

    paa_df = pd.read_csv(PAPER_AUTHOR_AFFILIATIONS_FILE, delimiter='\t', header=None,
                        names=["paper_id", "author_id", "affiliation_id",
                               "institution_full","institution", "number"])

    dataframe = pd.merge(pd.merge(paa_df, affiliations_df), papers_df)
    return dataframe


main()
