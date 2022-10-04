import token
import tensorflow as tf
from transformers import AutoTokenizer
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np

class Dataset:
    def __init__(self, max_len, model_path, max_code_cells, max_md_cells):
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_code_cells = max_code_cells
        self.max_md_cells = max_md_cells


    def preprocess_content(self, cell):
        """
        Preprocess the content of a cell by clean source, tokenizing and getting features.
        Args:
            content (str): The cell.
        Returns:
            content (str): Preprocessed content.
        """

        def clean_text(text):
            """
            Make the text cleaner.
            Args:
                text (str): text.
            Returns:
                text (str): A cleaned text.
            """
            
            text = str(text)
            text = text.lower().strip()
            text = re.sub(r"([?.!,¿])", r" \1 ", text)
            text = re.sub(r'[" "]+', " ", text)
            text = text.strip()

            return text


        def tokenize(sentence):
            """
            Tokenize a sentence.
            
            Args:
                sentence (str): A sentence.
            Returns:
                tokens (tuple): A list of input ids and attention mask.
            """
            
            tokens = self.tokenizer.encode_plus(
                sentence, 
                max_length=self.max_len,
                truncation=True, 
                padding='max_length',
                add_special_tokens=True, 
                return_attention_mask=True,
                return_token_type_ids=False, 
                return_tensors='tf'
            )
            
            return tokens['input_ids'], tokens['attention_mask']


        # Clean the source
        cell.source = clean_text(cell.source)

        # Tokenize the source
        input_ids, attention_mask = tokenize(cell.source)
        cell.input_ids = input_ids.numpy()[0].tolist()
        cell.attention_mask = attention_mask.numpy()[0].tolist()

        return cell


    def preprocess_dataset(self, df):
        """
        Preprocess the dataset for training.
        Args:
            df (pd): Dataset.
        Returns:
            dataset (pd): Preprocessed dataset.
        """

        df['input_ids'] = 0.
        df['attention_mask'] = 0.

        tqdm.pandas(desc="Preprocessing dataset")
        df = df.progress_apply(self.preprocess_content, axis=1)

        df = df.drop(['source', 'rank', 'ancestor_id'], axis=1)

        return df


    def get_notebook_token(self, df, max_cells, cell_pad):
        """
        Get the tokens for model. In this function, we'll pad the notebooks to be equal in term of number of cells that a notebook can maximum contains (i.e. This process is pretty much the same compared to the way we pad for a single sentence before). The returned cell_mask will tell us whether it's actually a real cell (real cell -> 1) or a padded version (fake cell -> 0).
        Args:
            df (pd): The tokenized notebook dataframe which contains the tokens instead of the rough content for each cell.
        Returns:
            input_ids (np array): Input ids with shape (num_notebooks, num_cells, max_len)
            attention_mask (np array): Attention mask with shape (num_notebooks, num_cells, max_len)
            cell_features (np array): Cell features with shape (num_notebooks, num_cells, 2)
            cell_mask (np array): Cell mask with shape (num_notebooks, num_cells, 1)
            target (np array): Percentile rank with shape (num_notebooks, num_cells, 1)
        """

        def create_tensor(col, desired_shape, dtype="int32"):
            """
            Create the desired tensor.
            Args:
                col (str): Column name needed to be tensorized.
                desired_shape (tuple): Desired output's shape.
                dtype (str): Data type. Default is int32.
            Returns:
                out (np array): Padded output with the shape of desired_shape.
            """

            out = np.full(shape=desired_shape, fill_value=cell_pad, dtype=dtype)
            
            count = 0
            for _, group in df.groupby("id"):
                value = group[col].tolist()
                value_shape = np.array(value).shape
                
                if len(value_shape) == 1:
                    out[count, :value_shape[0]] = value
                else:
                    out[count, :value_shape[0], :value_shape[1]] = value

                count += 1

            return out
        

        num_train = df.id.nunique()

        # input_ids
        input_ids = create_tensor(
            "input_ids", 
            (num_train, max_cells, self.max_len)
        )

        # attention_mask
        attention_mask = create_tensor(
            "attention_mask", 
            (num_train, max_cells, self.max_len)
        )
        
        # target
        target = create_tensor(
            "pct_rank", 
            (num_train, max_cells), 
            dtype="float32"
        )

        features = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        ids = list(df.groupby("id").groups.keys())

        return ids, features, target


    def build_dataset(self, batch_size, df=None, cell_pad=0., exceed_cells_action="filter"):
        """
        Build the dataset for training.
        Args:
            df (pd): Notebook dataframe. If provided, the dataset will be using. Otherwise, the dataset will be loaded from the disk.
        Returns:
            df (pd): Processed dataframe for reconstruction purpose. This dataset has only the notebook ids and cell ids columns.
            batched_set: Batched dataset for training.
            exceed_cells_action (str): The action to take when the number of cells in a notebook exceeds the maximum number of cells. There are two options: "filter" and "truncate": (1) "filter" (Default action) will filter out the notebooks that exceed the maximum number of cells, (2) "truncate" will truncate the some cells to ensure the notebooks do not have more than accepted number of cells.
        """

        def filter_by_num_cells(df):
            """
            Filter the notebooks by the number of cells containing.
            Args:
                df (pd): Notebook dataframe.
            Returns:
                filtered_df (pd): The dataframe after being filtered.
            """

            cell_count = df.groupby(["id", "cell_type"])['cell_id'].count().reset_index(name='counts')

            x = cell_count[(cell_count.cell_type == "code") & (cell_count.counts <= self.max_code_cells)].id
            x = set(x)
            y = cell_count[(cell_count.cell_type == "markdown") & (cell_count.counts <= self.max_md_cells)].id
            y = set(y)

            accepted_ids = x.intersection(y)
            accepted_ids = list(accepted_ids)

            filtered_df = df[df['id'].isin(accepted_ids)]
            filtered_df = filtered_df.sort_values(by=['id', 'cell_type'])
            return filtered_df


        def map_func(code_features, md_features, target):
            code_input_ids = code_features["input_ids"]
            code_attention_mask = code_features["attention_mask"]

            md_input_ids = md_features["input_ids"]
            md_attention_mask = md_features["attention_mask"]

            return ( 
                {
                    'code_input_ids': code_input_ids, 
                    'code_attention_mask': code_attention_mask, 
                    'md_input_ids': md_input_ids,
                    'md_attention_mask': md_attention_mask
                }, 
                target 
            )


        if exceed_cells_action == "filter":
            df = filter_by_num_cells(df)

        df = self.preprocess_dataset(df)
        
        code = df[df.cell_type == "code"]
        nb_ids_1, code_features, _ = self.get_notebook_token(code, self.max_code_cells, cell_pad)
        md = df[df.cell_type == "markdown"]
        nb_ids_2, md_features, target = self.get_notebook_token(md, self.max_md_cells, cell_pad)

        assert nb_ids_1 == nb_ids_2
        assert all(md.id.unique() == nb_ids_2)

        dataset = tf.data.Dataset.from_tensor_slices((
            code_features, 
            md_features, 
            target
        ))
        dataset = dataset.map(map_func)
        dataset = dataset.batch(batch_size)

        return md, dataset
        