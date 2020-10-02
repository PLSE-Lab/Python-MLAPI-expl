from collections import defaultdict
from itertools import combinations
import pandas as pd

# Confidence threshold
THRESHOLD = 0.5

# Only consider rules for items appearing at least `MIN_COUNT` times.
MIN_COUNT = 5

class pairwise_association_mining:
    def __init__(self, list_of_sets, threshold, min_count):
        assert isinstance(list_of_sets, list), "list_of_sets must be a list of sets"
        assert isinstance(list_of_sets[0], set), "list_of_sets must be a list of sets"
        assert isinstance(threshold, float) and threshold > 0  and threshold < 1, "threshold must be between 0 and 1"
        assert isinstance(min_count, int), "min_count must be an int"
        
        self.list_of_sets = list_of_sets
        self.threshold = threshold
        self.min_count = min_count
        
        self.pair_counts = defaultdict(int)
        self.item_counts = defaultdict(int)
        
        self.rules = dict()
        self.find_assoc_rules()
        
        self.pairwise_confidence = {pair:self.rules[pair] for pair in self.rules.keys() \
                             if self.item_counts[pair[0]] >= self.min_count}
        
    def update_pair_counts(self, itemset):
        """
        Updates a dictionary of pair counts for
        all pairs of items in a given itemset.
        """
        for a,b in combinations(itemset,2):
            self.pair_counts[(a,b)] += 1
            self.pair_counts[(b,a)] += 1
            
    def update_item_counts(self, itemset):
        """
        Updates a dictionary of item counts for
        all pairs of items in a given itemset.
        """
        for item in itemset:
            self.item_counts[item] += 1
            
    def filter_rules_by_conf(self):
        """
        Filters out pairs whose confidence is
        below the user defined threshold.
        """
        for (a,b) in self.pair_counts:
            confidence = self.pair_counts[(a,b)] / self.item_counts[a]
            if confidence >= self.threshold:
                self.rules[(a,b)] = confidence

    def find_assoc_rules(self):
        """
        Set final rules dictionary using
        pairs that appear together with
        confidence greater than or equal to
        the user defined threshold.
        """
        for itemset in self.list_of_sets:
            self.update_pair_counts(itemset)
            self.update_item_counts(itemset)
        rules = self.filter_rules_by_conf()
        return rules
    
    @staticmethod
    def gen_rule_str(a, b, val=None, val_fmt='{:.3f}', sep=" = "):
        text = "{} => {}".format(a, b)
        if val:
            text = "conf(" + text + ")"
            text += sep + val_fmt.format(val)
        return text

    def print_rules(self):
        """
        Pretty print pairwise associations
        """
        from operator import itemgetter
        ordered_rules = sorted(self.pairwise_confidence.items(), key=itemgetter(1), reverse=True)
        for (a, b), conf_ab in ordered_rules:
            print(self.gen_rule_str(a, b, conf_ab))
            
def main():
    df = pd.read_csv('../input/BreadBasket_DMS.csv')
    checkout_list = defaultdict(list)
    trans = dict()
    for row in df.groupby(by='Transaction').filter(lambda x: len(set(x['Item'])) > 1)[['Transaction','Item']].itertuples():
        if "{}".format(row.Transaction)+row.Item not in trans:
            checkout_list[row.Transaction].append(row.Item)
        trans["{}".format(row.Transaction)+row.Item] = None
        
    grocery_itemset = [set(lst) for lst in checkout_list.values()]
    pam = pairwise_association_mining(grocery_itemset, THRESHOLD, MIN_COUNT)
    pam.print_rules()
    
if __name__ == "__main__":
    main()