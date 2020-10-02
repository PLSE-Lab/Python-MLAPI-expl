import pandas as pd
from glob import glob
from tqdm import tqdm_notebook as tqdm
from dask import delayed

class SC_Data_Lake(object):

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.cat_filter_list = []
        self.add_cat_list = []

    def add_filter(self, cat_title, inst_cat_search):
        '''
        Add a single filtering step; e.g. filter for barcodes with
        this category of this category type.

        To do: support exclusion of category by adding third element to tuple.
        '''
        if type(inst_cat_search) is list:
            inst_cat_search = tuple(inst_cat_search)
        self.cat_filter_list.append((cat_title, inst_cat_search))

    def reset_filters(self):
        self.cat_filter_list = []

    def add_cats(self, add_cat_list):
        '''
        Add list of categories that will be added to the search results (barcodes
        that meet the search criteria).
        '''
        self.add_cat_list = add_cat_list

    def add_cats_from_meta(self, barcodes, df_meta, add_cat_list):
        '''
        Add categories from df_meta.
        '''

        # get metadata of interest (add_cat_list) from barcodes of interest
        df_cats = df_meta.loc[barcodes][add_cat_list]

        # get list of cats
        list_cat_ini = [list(x) for x in df_cats.values]

        # add titles to cats
        list_cat_titles = [ list([str(x) + ': ' + str(y) for x,y in zip(add_cat_list, a)]) for a in list_cat_ini]

        # add barcodes to new columns
        new_cols = [tuple([x] + y) for x,y in zip(barcodes, list_cat_titles)]

        return new_cols

    def get_samples(self, test_run=False):
        '''
        Get all samples from the base direcotry
        '''

        if test_run != False:
            print('*** test run: only working with subset of samples')
            all_samples = sorted(glob(self.base_dir + '*'))[:test_run]
        else:
            all_samples = sorted(glob(self.base_dir + '*'))

        return all_samples

    def get_cat_distribution_across_samples(self, inst_cat_type, test_run=False):

        all_samples = self.get_samples(test_run)

        ser_list = []
        for inst_sample in tqdm(all_samples):

            sample_name = inst_sample.split('/')[-1]
            meta_file = sample_name + '_cell-metadata.parquet'

            # load_metadata
            df_meta = pd.read_parquet(inst_sample + '/' + meta_file)

            # get series of category
            cat_ser = df_meta[inst_cat_type]

            # get value counts of that category
            cat_counts = cat_ser.value_counts()
            cat_counts.name = sample_name

            ser_list.append(cat_counts)

        # need to set up pandas version 0.22 vs 0.24 sort=True option
        df_dist = pd.concat(ser_list, axis=1)

        return df_dist

    def report_cat_filter_list(self):
        for cat_tuple in self.cat_filter_list:
            inst_cat_type = cat_tuple[0]
            inst_cat_search = cat_tuple[1]

            print(inst_cat_type, ' == ', inst_cat_search)

    def filter_meta_using_cat_filter_list(self, df_meta):

        for cat_tuple in self.cat_filter_list:
            inst_cat_type = cat_tuple[0]
            inst_cat_search = cat_tuple[1]

            if type(inst_cat_search) is not tuple:
                # find indexes of barcodes that match requested caetgory
                inst_cat = inst_cat_search
                cat_ser = df_meta[inst_cat_type]
                found_barcodes = cat_ser[cat_ser == inst_cat].index.tolist()
            else:
                # find indexes of barcodes that match requested categories
                found_barcodes = []
                for inst_cat in inst_cat_search:
                    cat_ser = df_meta[inst_cat_type]
                    inst_found = cat_ser[cat_ser == inst_cat].index.tolist()
                    found_barcodes.extend(inst_found)

            # apply progressive filters to metadata
            df_meta = df_meta.loc[found_barcodes]

        return df_meta

    def get_sample_gex(self, inst_sample, make_sparse=True):
        sample_name = inst_sample.split('/')[-1]
        meta_file = sample_name + '_cell-metadata.parquet'

        df_meta = pd.read_parquet(inst_sample + '/' + meta_file)

        df_meta = self.filter_meta_using_cat_filter_list(df_meta)

        found_barcodes = df_meta.index.tolist()

        df_gex = pd.SparseDataFrame()

        # load gene expression data
        if len(found_barcodes) > 0:
            gex_file = sample_name + '.parquet'

            df_gex = pd.read_parquet(inst_sample + '/' + gex_file, columns=found_barcodes)

            if make_sparse:
                df_gex = pd.SparseDataFrame(df_gex, default_fill_value=0)

            # add categories to barcodes
            df_gex.columns = self.add_cats_from_meta(df_gex.columns.tolist(), df_meta, self.add_cat_list)

        return df_gex

    def run_cell_search(self, make_sparse=True, test_run=False):

        all_samples = self.get_samples(test_run)

        df_list = []
        for inst_sample in tqdm(all_samples):

            df_gex = self.get_sample_gex(inst_sample, make_sparse)

            if df_gex.shape[1] > 0:
                df_list.append(df_gex)

        df_merge_gex = pd.concat(df_list, axis=1)

        return df_merge_gex

    def dask_run_cell_search(self, make_sparse=True, test_run=False):
        print('dask_run_cell_search')
        self.report_cat_filter_list()

        all_samples = self.get_samples(test_run)

        df_list = []
        for inst_sample in all_samples:

            df_gex = delayed(self.get_sample_gex)(inst_sample, make_sparse)

            df_list.append(df_gex)

        df_merge_gex = delayed(pd.concat)(df_list, axis=1).compute()

        return df_merge_gex

    def get_genes_from_all_samples(self, gene_list, make_sparse=True, test_run=False):

        self.report_cat_filter_list()

        if type(gene_list) is not list:
            gene_list = [gene_list]

        all_samples = self.get_samples(test_run)

        df_list = []
        for inst_sample in tqdm(all_samples):

            df_gene = self.get_df_gene(inst_sample, gene_list, make_sparse)

            df_list.append(df_gene)

        df_merge_gene = pd.concat(df_list, axis=0)

        return df_merge_gene

    def dask_get_genes_from_all_samples(self, gene_list, make_sparse=True, test_run=False):

        self.report_cat_filter_list()

        if type(gene_list) is not list:
            gene_list = [gene_list]

        all_samples = self.get_samples(test_run)

        df_list = []
        for inst_sample in all_samples:

            df_gene = delayed(self.get_df_gene)(inst_sample, gene_list, make_sparse)

            df_list.append(df_gene)

        df_merge_gene = delayed(pd.concat)(df_list, axis=0).compute()

        return df_merge_gene

    def get_df_gene(self, inst_sample, gene_list, make_sparse=True):

        sample_name = inst_sample.split('/')[-1]
        gene_file = sample_name + '_gene-column.parquet'
        meta_file = sample_name + '_cell-metadata.parquet'

        # load_metadata
        df_meta = pd.read_parquet(inst_sample + '/' + meta_file)

        # load gene data
        df_gene = pd.read_parquet(inst_sample + '/' + gene_file , columns=gene_list)

        if make_sparse:
            df_gene = pd.SparseDataFrame(df_gene, default_fill_value=0)

        if len(self.cat_filter_list) > 0:
            # Filter for barcodes/cells based on filter_cat_list
            df_meta = self.filter_meta_using_cat_filter_list(df_meta)

            # only keep genes that meet filtering criteria
            df_gene = df_gene.loc[df_meta.index.tolist()]

        if len(self.add_cat_list) > 0:
            # add categories to barcodes
            df_gene.index = self.add_cats_from_meta(df_gene.index.tolist(), df_meta, self.add_cat_list)

        return df_gene