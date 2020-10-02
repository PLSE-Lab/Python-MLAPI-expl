import re
import os
import pandas as pd
import numpy as np
from google.cloud import storage, automl_v1beta1 as automl

class AutoMLTablesWrapper():
    def __init__(self, project_id, bucket_name, dataset_display_name, train_filepath, 
                 test_filepath, target_column, id_column, model_display_name,
                 train_budget, bucket_region='us-central1', clean_train_filepath='train.csv', 
                 clean_test_filepath='test.csv'):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.dataset_display_name = dataset_display_name
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.target_column = target_column
        self.id_column = id_column
        self.model_display_name = model_display_name
        self.train_budget = train_budget
        self.bucket_region = bucket_region
        self.clean_train_filepath = clean_train_filepath
        self.clean_test_filepath = clean_test_filepath
        
        self.everything_looks_good()
        self.clean_data(train_filepath, test_filepath)
        self.prepare_clients()
        self.create_gcs_bucket()
        self.upload_blob(clean_train_filepath, clean_train_filepath)
        self.upload_blob(clean_test_filepath, clean_test_filepath)
        self.find_or_create_dataset()
        self.set_target_column()
        self.make_columns_nullable()
        print('Ready to train model.')
        
    def everything_looks_good(self):
        assert type(self.project_id) == str, "PROJECT_ID should be a Python string."
        assert type(self.bucket_name) == str, "BUCKET_NAME should be a Python string."
        assert type(self.dataset_display_name) == str, "DATASET_DISPLAY_NAME should be a Python string."
        assert type(self.train_filepath) == str, "TRAIN_FILEPATH should be a Python string."
        assert type(self.test_filepath) == str, "TEST_FILEPATH should be a Python string."
        assert type(self.target_column) == str, "TARGET_COLUMN should be a Python string." 
        assert type(self.id_column) == str, "ID_COLUMN should be a Python string."
        assert type(self.model_display_name) == str, "MODEL_DISPLAY_NAME should be a Python string."
        assert type(self.train_budget) == int, "TRAIN_BUDGET should be an integer."
        # check train_filepath and test_filepath are CSV files
        assert os.path.exists(self.train_filepath), "Could not find a file at the value provided for TRAIN_FILEPATH."
        assert os.path.exists(self.test_filepath), "Could not find a file at the value provided for TEST_FILEPATH."
        assert self.train_filepath.endswith("csv"), "The training dataset needs to be a CSV file."
        assert self.test_filepath.endswith("csv"), "The test dataset needs to be a CSV file."
        # check allowed length, allowed characters in display names
        assert len(self.model_display_name) <= 32, "MODEL_DISPLAY_NAME should use at most 32 characters." 
        assert len(self.dataset_display_name) <= 32, "DATASET_DISPLAY_NAME should use at most 32 characters."
        assert self.allowed_characters(self.model_display_name), "MODEL_DISPLAY_NAME contains a character that is not allowed. The only allowed characers are ASCII Latin letters A-Z and a-z, an underscore (_), and ASCII digits 0-9."
        assert self.allowed_characters(self.dataset_display_name), "DATASET_DISPLAY_NAME contains a character that is not allowed. The only allowed characers are ASCII Latin letters A-Z and a-z, an underscore (_), and ASCII digits 0-9."
        # check train_budget
        assert self.train_budget >= 1000 and self.train_budget <= 72000, "TRAIN_BUDGET must be a value between 1000 and 72000, inclusive."
        
    def allowed_characters(self, strg, search=re.compile(r'[^\_a-z0-9]').search):
        return not bool(search(strg.lower()))
    
    def clean_data(self, train_filepath, test_filepath):
        # Step 1A: Load the CSV files into DataFrames
        train_df = pd.read_csv(train_filepath)
        test_df = pd.read_csv(test_filepath)
        # Step 1B: Get list of columns to make nullable
        self.nullable_columns = list(set(train_df.columns[train_df.isnull().any()]) | set(test_df.columns[test_df.isnull().any()]))
        # Step 2: Replace missing values with empty string
        cleaned_train_df = train_df.replace(np.nan, "")
        cleaned_test_df = test_df.replace(np.nan, "")
        # Step 3: Saves cleaned DataFrames to '/kaggle/working/train.csv' and '/kaggle/working/test.csv'
        cleaned_train_df.to_csv(path_or_buf=self.clean_train_filepath, index=False)
        cleaned_test_df.to_csv(path_or_buf=self.clean_test_filepath, index=False)
        # Step 4: Check target column and ID column are in dataset
        all_columns = list(cleaned_train_df.columns)
        assert self.id_column in all_columns, "ID_COLUMN not a valid column name."
        assert self.target_column in all_columns, "TARGET_COLUMN not a valid column name."
    
    def prepare_clients(self):
        print('Preparing clients ...')
        # create storage client
        self.storage_client = storage.Client(project=self.project_id)
        # create tables client
        automl_client = automl.AutoMlClient()
        tables_gcs_client = automl.GcsClient(client=self.storage_client, bucket_name=self.bucket_name)
        prediction_client = automl.PredictionServiceClient()
        self.tables_client = automl.TablesClient(project=self.project_id, region=self.bucket_region, 
                                                 client=automl_client, gcs_client=tables_gcs_client, prediction_client=prediction_client)
        print('Clients successfully created!')
        
    def create_gcs_bucket(self):
        bucket = storage.Bucket(self.storage_client, name=self.bucket_name)
        if not bucket.exists():
            bucket.create(location=self.bucket_region)
            print('GCS bucket created.')
        else:
            print('GCS bucket found.')
            
    def upload_blob(self, source_file_name, destination_blob_name):
        """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """
        bucket = self.storage_client.get_bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print('File {} uploaded to {}.'.format(
            source_file_name,
            destination_blob_name))
        
    def find_or_create_dataset(self):
        try:
            self.dataset = self.tables_client.get_dataset(dataset_display_name=self.dataset_display_name)
            print('Dataset found.')
        except:
            print('Creating dataset ...')
            self.dataset = self.tables_client.create_dataset(self.dataset_display_name)
            
            # need to fix later once can demonstrate it works
            gcs_input_uris = ['gs://' + self.bucket_name + '/' + self.clean_train_filepath]
            import_data_operation = self.tables_client.import_data(
                dataset=self.dataset,
                gcs_input_uris=gcs_input_uris
            )
            # Synchronous check of operation status. Wait until import is done.
            import_data_operation.result()
            print('Dataset successfully created!')
            
    def set_target_column(self):
        self.tables_client.set_target_column(dataset=self.dataset, column_spec_display_name=self.target_column)
        print('Set target column.')
        
    def make_columns_nullable(self):
        for col in self.tables_client.list_column_specs(self.project_id, self.bucket_region, self.dataset.name):
            if self.target_column == col.display_name or self.id_column == col.display_name:
                continue
            if col in self.nullable_columns:
                self.tables_client.update_column_spec(self.project_id,
                                                  self.bucket_region,
                                                  self.dataset.name,
                                                  column_spec_display_name=col.display_name,
                                                  type_code=col.data_type.type_code,
                                                  nullable=True)
        print('Set columns to nullable.')
            
    def train_model(self):
        self.model = None
        try:
            self.model = self.tables_client.get_model(model_display_name=self.model_display_name)
        except:
            response = self.tables_client.create_model(
                self.model_display_name,
                dataset=self.dataset,
                train_budget_milli_node_hours=self.train_budget,
                exclude_column_spec_names=[self.id_column, self.target_column]
            )
            response.operation
            print('Training model ...')
            # Wait until model training is done.
            self.model = response.result()
            print('Finished training model.')

    def download_to_kaggle(self, destination_directory, file_name, prefix=None):
        """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""
        os.makedirs(destination_directory, exist_ok=True)
        full_file_path = os.path.join(destination_directory, file_name)
        blobs = self.storage_client.list_blobs(self.bucket_name, prefix=prefix)
        for blob in blobs:
            blob.download_to_filename(full_file_path)
            
    def get_predictions(self, prediction_filepath='tables_1.csv'):
        # calculate predictions
        gcs_input_uris = 'gs://' + self.bucket_name + '/' + self.clean_test_filepath
        gcs_output_uri_prefix = 'gs://' + self.bucket_name + '/predictions'

        print('Getting predictions ...')
        batch_predict_response = self.tables_client.batch_predict(
            model=self.model, 
            gcs_input_uris=gcs_input_uris,
            gcs_output_uri_prefix=gcs_output_uri_prefix,
        )
        batch_predict_response.operation
        # Wait until batch prediction is done.
        batch_predict_result = batch_predict_response.result()
        
        # download predictions to kaggle
        gcs_output_folder = batch_predict_response.metadata.batch_predict_details.output_info.gcs_output_directory.replace('gs://' + self.bucket_name + '/','')
        self.download_to_kaggle('/kaggle/working', prediction_filepath, prefix=gcs_output_folder)
        
        # format submission
        preds_df = pd.read_csv(prediction_filepath)
        submission_df = preds_df[[self.id_column, 'predicted_' + self.target_column]]
        submission_df.columns = [self.id_column, self.target_column]
        submission_df.to_csv('submission.csv', index=False)
        print("Submission ready for download!")