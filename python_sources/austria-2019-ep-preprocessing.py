
import pandas as pd

FILENAME = ("/kaggle/input/austrian-ep-elections-data-2019/"
            "endgueltiges_Ergebnis_inkl_WK_mit_Gemeindeergebnissen_EW19.csv")

PARTY_NAMES = [
    'ÖVP', 'SPÖ', 'FPÖ', 'GRÜNE',
    'NEOS', 'KPÖ', 'EUROPA'
]


class DataColumnsInfo():
    def __init__(self):
        self._parties = PARTY_NAMES.copy()

    @property
    def parties(self):
        return self._parties

    @property
    def area_code(self):
        return "GKZ"

    @property
    def valid_votes(self):
        return "Stimmen,gültige"

    @property
    def registered_voters(self):
        return "Wahl-berechtigte"

    def __repr__(self):
        return (
            "DataColumnsInfo(\n"
            "    area_code: '%s',\n"
            "    valid_votes: '%s',\n"
            "    registered: '%s',\n"
            "    parties: %s,\n"
            ")" %
            (self.area_code, self.valid_votes,
             self.registered_voters, self.parties)
        )


def get_cleaned_data():
    """ Obtain the cleaned data - meaning that only the
        per electoral ward data is left in, aggregates
        and postal votes (briefwahl) filtered out.
    """
    aggregate_code_end = "00"
    postal_vote_code_end = "99"

    df = pd.read_csv(FILENAME)

    df = df[~df.GKZ.astype(str).str.endswith(aggregate_code_end) &
            ~df.GKZ.astype(str).str.endswith(postal_vote_code_end)]
    df.reset_index(inplace=True)
    df.drop(columns=["index"], inplace=True)

    cols = list(df.columns)
    stimmen = cols[3]
    cols[3:6] = [stimmen + "," + x for x in df.iloc[0, 3:6]]
    df.columns = cols

    df.drop([0], inplace=True)

    df.index = range(len(df))
    info = DataColumnsInfo()

    df[info.valid_votes] = df[info.valid_votes].astype(int)
    for col in PARTY_NAMES:
        df[col] = df[col].astype(int)
    return df, info


def get_preprocessed_data(remove_perc_cols=True):
    """ Load the Austrian 2019 EP Elections data, and remove subtotals and postal 
        votes. This is beneficial when trying to detect manipulations taking place
        before adding up these numbers to get total votes, as well as excluding 
        subsequent correlated digits of those areas which consist of a single
        electoral ward only (such per ward counts would be followed by an equal 
        total value). 
        
        :param remove_perc_cols: Whether to remove vote percentages.
        :return: The preprocessed data frame and a DataColumnsInfo() instance
            providing details about the structure (just print it).
            
        Example:
        df, info = get_preprocessed_data()
    """
    df, info = get_cleaned_data()
    if remove_perc_cols:
        df = df[[col for col in df.columns
                 if not col.startswith("%")]]

    for col in PARTY_NAMES:
        df["ld_" + col] = df[col] % 10
    return df, info


if __name__ == "__main__":
    df, info = get_preprocessed_data()
    print(df.head())
    print(info)
