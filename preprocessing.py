"""
preprocessing.py  --  build the per-database schema lookup for the NL2SQL agent.

For every Spider database this builds and stores:
    col_dicts   : list of per-column metadata dicts
    embeddings  : (n_cols, 384) matrix; row i belongs to col_dicts[i]
    graph       : networkx graph over columns (FK edges + same-table edges)
    schema      : short "Tables / ForeignKeys" string shown to the agent

Everything is saved to one pickle (spider_schema_lookup.pkl) that runtime_tools.py loads.

Requirements:
    pip install datasets sentence-transformers scikit-learn networkx numpy

Environment variables (optional):
    SPIDER_DB_PATH   path to Spider's `database/` folder (the one holding <db_id>/<db_id>.sqlite)
    SCHEMA_PKL_PATH  where to write the pickle (default: spider_schema_lookup.pkl)

Usage:
    python preprocessing.py
"""

import os
import pickle
import sqlite3
from itertools import combinations
from collections import defaultdict

import numpy as np
import networkx as nx
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# DeepSeek has no embeddings endpoint, so column embeddings are computed locally.
# runtime_tools.py MUST use the same model, otherwise query vectors and stored vectors
# live in different spaces.
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

DB_BASE_PATH = os.environ.get(
    'SPIDER_DB_PATH',
    '/kaggle/input/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset/spider/database',
)
OUTPUT_PATH = os.environ.get('SCHEMA_PKL_PATH', 'spider_schema_lookup.pkl')

# Load once globally, not inside the function
model = SentenceTransformer(EMBEDDING_MODEL)

# filled in main(); parse_schema() reads from it
schema_lookup = {}


# ----------------------------------------------------------------------------
# utils: parsing functions
# ----------------------------------------------------------------------------

def parse_primary_keys(pk_string):

  pk_map ={}
  # There are database full of tables but has not primary keys at all.
  if not pk_string:
    return pk_map

  for entry in pk_string.split('|'):
    entry = entry.strip()
    table, cols = entry.split(':')

    pk_map[table.strip()] =cols.strip()
  return pk_map


def parse_foreign_keys(fk_string):
    fk_map = {}
    words = fk_string
    fk_string = words.split()
    for i,word in enumerate(fk_string):
        if word =='equals':
            first_table = fk_string[i-3]
            second_table = fk_string[i+1]

            fk_map[(first_table, fk_string[i-1])] = (second_table, fk_string[i+3])


    return fk_map


# ----------------------------------------------------------------------------
# Creating col level metadata for each db
# ----------------------------------------------------------------------------

def parse_schema(db_id):
    """
    Parses one database's schema strings into a list of column dicts.

    At this stage each dict has:
        'table_name', 'column_name', 'column_type',
        'is_primary_key': bool,
        'foreign_key': None or {'ref_table': ..., 'ref_column': ...}

    Two more keys are added later by other functions:
        'statistics' <- compute_statistics()
        'embed_text' <- compute_embeddings()
    """

    row = schema_lookup[db_id]
    schema = row['Schema (values (type))']
    pk_map = parse_primary_keys(row['Primary Keys'])
    fk_map = parse_foreign_keys(row['Foreign Keys'])

    col_dicts = []

    for table_info in schema.split('|'):

        table_info = table_info.strip()

        table_name, cols_info = table_info.split(':')
        table_name = table_name.strip()
        cols_info = cols_info.strip()

        # a table can have no primary key.
        pk =''
        if table_name in pk_map:
            pk = pk_map[table_name]

        for col_info in cols_info.split(','):

            col_info = col_info.strip()

            col_name = col_info[:col_info.rfind('(')].strip()
            col_type = col_info[col_info.rfind('(')+1 : col_info.rfind(')')].strip()

            # checking if foreign key exist for this column
            fk_found = False
            col_to_find = col_name

            if (table_name, col_to_find) in fk_map:
                mapping = fk_map[(table_name, col_to_find)]
                fk_found = True

            temp_dic ={}
            temp_dic['table_name'] = table_name
            temp_dic['column_name'] = col_name
            temp_dic['column_type'] = col_type
            temp_dic['is_primary_key'] = True if pk and col_name == pk else False
            temp_dic['foreign_key'] = None

            if fk_found:

                temp = {}
                temp['ref_table'] = mapping[0]
                temp['ref_column'] = mapping[1]
                temp_dic['foreign_key'] = temp

            col_dicts.append(temp_dic)
    return col_dicts


# ----------------------------------------------------------------------------
# Computing statistics for each column of all tables in db
# ----------------------------------------------------------------------------

def compute_statistics(col_dicts, db_id, db_base_path):
    """
    Takes col_dicts from parse_schema, connects to the SQLite DB,
    and adds a 'statistics' field to each dict in place.
    The statistics are the first 3 distinct non-null values of the column.

    db_base_path: path to the 'database' folder
    e.g: '/kaggle/input/spider/database'
    """
    db_path = os.path.join(db_base_path, db_id, f"{db_id}.sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for col in col_dicts:
        table = col['table_name']
        column = col['column_name']

        try:
            cursor.execute(
                f"SELECT DISTINCT `{column}` FROM `{table}` "  #Backtick quoting around column and table names — necessary because some Spider column names are reserved SQL words (Year, Name, Average) that will throw a syntax error without quoting
                f"WHERE `{column}` IS NOT NULL LIMIT 3"
            )
            rows = cursor.fetchall()
            values = [row[0] for row in rows]
            col['statistics'] = f"e.g: {values}"

        except Exception as e:
            col['statistics'] = "N/A"

    conn.close()
    return col_dicts


# ----------------------------------------------------------------------------
# Creating schema (table + fk relationship) for each db (broader topology for model to traverse)
# ----------------------------------------------------------------------------

def build_schema(col_dicts):

    schema = ''

    check_table= {}
    table_str = ''
    Fk_str = ''

    for col_meta in col_dicts:

        table_name = col_meta['table_name']
        col_name = col_meta['column_name']

        if table_name not in check_table:

            check_table[table_name] = 1

            if not table_str:
                table_str+= table_name
            else:
                table_str+= f', {table_name}'


        if col_meta['foreign_key']:

            ref_table = col_meta['foreign_key']['ref_table']
            ref_column = col_meta['foreign_key']['ref_column']

            text = f'{table_name}.{col_name} = {ref_table}.{ref_column}'

            if not Fk_str:
                Fk_str+= text
            else:
                Fk_str+= f'\n{text}'

    schema = f'Tables: {table_str}\nForeignKeys:\n{Fk_str}'
    return schema


# ----------------------------------------------------------------------------
# Computing column embeddings for each db
# ----------------------------------------------------------------------------

def compute_embeddings(col_dicts):
    """
    Takes col_dicts (with statistics already computed).
    Adds 'embed_text' to each dict.
    Returns embedding matrix of shape (n_cols, 384), L2-normalised.
    Row i of the matrix belongs to col_dicts[i].
    """

    text_to_embed = []
    for col in col_dicts:

        col_name = col['column_name'].replace('_', ' ').lower() #Singer_ID as "singer id"
        table_name = col['table_name'].replace('_', ' ').lower()

        embed_text = f"a column named {col_name} in table {table_name}."
        col['embed_text'] = embed_text

        text_to_embed.append(embed_text)

    embeddings = model.encode(text_to_embed)   # same order as input
    embeddings_norm = normalize(embeddings, norm='l2')

    return embeddings_norm


# ----------------------------------------------------------------------------
# Building graph for each db
# ----------------------------------------------------------------------------

def build_graph(col_dicts):

    G = nx.Graph()

    table_to_column_map = defaultdict(list)

    for col_dic in col_dicts:

        node = f"{col_dic['table_name']}.{col_dic['column_name']}" # olumns with no FK and are the only column in their table won't get added via add_edge. For safety added
        G.add_node(node)

        table = col_dic['table_name']
        col = col_dic['column_name']

        if col_dic['foreign_key']:

            ref_table = col_dic['foreign_key']['ref_table']
            ref_column = col_dic['foreign_key']['ref_column']

            start = f'{table}.{col}'
            end = f'{ref_table}.{ref_column}'

            G.add_edge(start, end)


        table_to_column_map[table].append(col)


    for table, col_list in table_to_column_map.items():

        pairs = combinations(col_list, 2)

        for pair in pairs:

            start = f'{table}.{pair[0]}'
            end = f'{table}.{pair[1]}'

            G.add_edge(start, end)

    return G


# ----------------------------------------------------------------------------
# main: computing for all db, then saving
# ----------------------------------------------------------------------------

def main():

    if not os.path.isdir(DB_BASE_PATH):
        raise SystemExit(
            f"Spider database folder not found: {DB_BASE_PATH}\n"
            "Set the SPIDER_DB_PATH environment variable to Spider's 'database' folder."
        )

    # loading the schema of the dbs
    spider_tables = load_dataset("richardr1126/spider-schema", split="train")
    print(spider_tables)

    for row in spider_tables:
        schema_lookup[row['db_id']] = row

    for i, (db_id, values) in enumerate(schema_lookup.items(), 1):

        col_dicts = parse_schema(db_id)
        col_dicts = compute_statistics(col_dicts, db_id, db_base_path = DB_BASE_PATH)
        embeddings = compute_embeddings(col_dicts)
        schema = build_schema(col_dicts)

        schema_lookup[db_id]['embeddings'] = embeddings
        schema_lookup[db_id]['col_dicts'] = col_dicts
        schema_lookup[db_id]['graph'] = build_graph(col_dicts)
        schema_lookup[db_id]['schema'] = schema

        print(f"[{i}/{len(schema_lookup)}] {db_id}: {len(col_dicts)} columns")

    # Pickle to disk
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(schema_lookup, f)

    print(f"\nSaved schema_lookup with {len(schema_lookup)} DBs to {OUTPUT_PATH}")

    # sanity checks
    db = schema_lookup['concert_singer']
    G = db['graph']

    print("\nconcert_singer embeddings:", db['embeddings'].shape)
    print("Nodes:", G.number_of_nodes())  # should be 21
    print("Edges:", G.number_of_edges())

    print("\nFK edges:")
    for u, v in G.edges():
        # FK edges cross tables
        if u.split('.')[0] != v.split('.')[0]:
            print(f"  {u} <-> {v}")

    print("\n" + schema_lookup['car_1']['schema'])


if __name__ == "__main__":
    main()
