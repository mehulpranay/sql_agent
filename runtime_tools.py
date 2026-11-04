"""
runtime_tools.py  --  the four tools the NL2SQL agent calls at runtime.

    search_column      semantic search over a database's columns (embeddings)
    search_value       keyword (LIKE) search over the actual cell values
    FindShortestPath   join path between columns, from the column graph
    ExecuteSQL         run a query, return the rows or the error message

All of them work on the pickle written by preprocessing.py
(spider_schema_lookup.pkl) and on Spider's SQLite files.

Requirements:
    pip install sentence-transformers networkx numpy

Environment variables (optional):
    SPIDER_DB_PATH   path to Spider's `database/` folder
    SCHEMA_PKL_PATH  path to spider_schema_lookup.pkl (default: ./spider_schema_lookup.pkl)

Usage:
    from runtime_tools import search_value, search_column, FindShortestPath, ExecuteSQL
    python runtime_tools.py        # runs a few demo calls
"""

import os
import sqlite3
import itertools
from itertools import combinations
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

# Must be the same model preprocessing.py used to build the stored column embeddings.
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

DB_BASE_PATH = os.environ.get(
    'SPIDER_DB_PATH',
    '/kaggle/input/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset/spider/database',
)
SCHEMA_PKL_PATH = os.environ.get('SCHEMA_PKL_PATH', 'spider_schema_lookup.pkl')

# Load once globally, not inside the function
model = SentenceTransformer(EMBEDDING_MODEL)


def search_column(query, col_dicts, embeddings, topk=5):
    """
    query: list of str — semantic descriptions of needed columns
    col_dicts: list of dicts for this DB (from schema_lookup)
    embeddings: np.array (n_cols, 384) for this DB, row i belongs to col_dicts[i]
    topk: number of results per query term

    Queries are embedded with the module-level `model` (same one used in preprocessing.py).

    Returns: dict {query_term: [result, ...]} for several terms, or the list directly if
    there is a single term. Each result is {'column', 'format', 'table', 'statistics'}.
    """

    if not query:
        return 'Nothing was sent to search'

    results = {}

    for q in query:
        q_norm = q.strip().lower()

        # Match their query transformation
        desc = f"a column about {q_norm}."

        # Encode query
        query_vec = model.encode(desc)   # numpy array
        query_vec_norm = query_vec / np.linalg.norm(query_vec)


        # Cosine similarity
        scores = embeddings @ query_vec_norm  # (n_cols,)
        # print(score.shape)

        # Exact name match override — force to top
        for i, col in enumerate(col_dicts):
            if q_norm == col['column_name'].lower():
                scores[i] = 1.0  # max possible cosine sim

        # Rank and take top-k
        top_indices = scores.argsort()[::-1][:topk]

        # Build result list
        format_results = []
        for idx in top_indices:
            col = col_dicts[idx]

            # Truncate statistics
            stats = col.get('statistics', 'N/A')
            if len(str(stats)) > 200:
                stats = str(stats)[:200] + f"...(Omit {len(str(stats))-200} chars)"

            format_results.append({
                'column': col['column_name'],
                'format': col['column_type'],
                'table': col['table_name'],
                'statistics': stats
            })

        results[q] = format_results

    # Single query → return list directly (match their behavior)
    if len(results) == 1:
        return results[query[0]]

    return results


def search_value(query, db_id, col_dicts, db_base_path, table = None, column = None):

    """
    query: str — value to search for e.g. "Soccer activity"
    db_id: str — which DB to search in
    col_dicts: list of dicts from schema_lookup
    db_base_path: path to database/ folder
    table: optional — narrow search to this table
    column: optional — narrow search to this column

    Returns: list of {'contents': ..., 'table': ..., 'column': ...}
    """

    # Checking if the table and column passed by the model is present in the db, or it hallucinated

    table_presence = False
    column_presence = False

    if table:
        for col_dic in col_dicts:
            if col_dic['table_name'] == table:
                table_presence = True
    if column:
        for col_dic in col_dicts:
            if col_dic['column_name'] == column:
                column_presence = True
    if table and not table_presence:
        return f'{table} named table not present in schema'
    if column and not column_presence:
        return f'{column} named column not present in schema'



    # list of stop words to not check since we tokenize and check it

    STOP_WORDS = {'the', 'a', 'an', 'of', 'in', 'at', 'by', 'for',
                  'with', 'about', 'activity', 'area', 'type', 'name'}

    # Connect to DB
    db_path = os.path.join(db_base_path, db_id, f"{db_id}.sqlite")
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors='ignore')
    cursor = conn.cursor()

    # Tokenize
    tokens = query.lower().split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    if not tokens:
        tokens = [query]  # fallback if all words were stop words


    list_of_contents = []
    search_list = []
    seen = set() # for deduplication.  Same (contents, table, column) can appear multiple times if multiple tokens match the same cell


    if table and column:
        search_list.append((table,column))


    elif table and not column:
        for col_dic in col_dicts:
            if col_dic['table_name'] == table:
                search_list.append((table, col_dic['column_name']))


    elif not table and not column:
        for col_dic in col_dicts:
            if col_dic['column_type'] in ('text', 'others'):
                search_list.append((col_dic['table_name'], col_dic['column_name']))

    for token in tokens:
        for search_window in search_list:
            t = search_window[0]
            c = search_window[1]

            try:
                cursor.execute(
                    f"SELECT DISTINCT `{c}` FROM `{t}` "
                    f"WHERE `{c}` LIKE ? AND `{c}` IS NOT NULL",
                    (f'%{token}%',)
                )
                rows = cursor.fetchall()
                for row in rows:

                    content = row[0]
                    key = (str(content), t, c)
                    if key not in seen:
                        seen.add(key)

                        temp = {}
                        temp['contents'] = content
                        temp['table'] = t
                        temp['column'] = c
                        list_of_contents.append(temp)
            except Exception as e:
                print(f"[search_value] query failed on {t}.{c}: {e}")
                continue

    conn.close()

    if not list_of_contents:
        return "No matching values found."

    return list_of_contents


def FindShortestPath(db_id, schema_lookup, start, end):

    if not start or not end:
        return "start or end list is empty"


    G = schema_lookup[db_id]['graph']

    pairs = list(itertools.product(start, end)) # [('Student.Fname', 'Activity.activity_name'), ('Faculty.Fname', 'Activity.activity_name')]

    list_of_paths = []
    for pair in pairs:

        source = pair[0]
        target = pair[1]

        try:
            path = nx.shortest_path(G, source, target)

        except nx.NetworkXNoPath:

            list_of_paths.append((source, target, "no path found"))
            continue

        except nx.NodeNotFound:

            missing = []
            if source not in G:
                missing.append(source)
            if target not in G:
                missing.append(target)
            msg = f"{' and '.join(missing)} not found in schema"
            list_of_paths.append((source, target, msg))
            continue

        temp = path[0]

        for i in range(1, len(path)):

            curr_table = path[i].split('.')[0]
            prev_table = path[i-1].split('.')[0]

            if curr_table == prev_table:
                temp+= f' <-> {path[i]}'
            else:
                temp+= f' = {path[i]}'

        list_of_paths.append((source, target, temp))

    return list_of_paths


def ExecuteSQL(sql: str, db_id: str, db_base_path: str):

    # Connect to DB
    db_path = os.path.join(db_base_path, db_id, f"{db_id}.sqlite")
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors='ignore')
    cursor = conn.cursor()

    try:
        cursor.execute(sql
        )
        rows = cursor.fetchall()
        return ('Successful',rows)

    except Exception as e:
        error_msg = str(e)
        return ( 'Unsuccessful',error_msg)

    finally:
        conn.close() # always executes regardless of return or exception


if __name__ == "__main__":
    # demo calls on the saved schema lookup
    import pickle

    with open(SCHEMA_PKL_PATH, 'rb') as file:
        schema_lookup = pickle.load(file)

    print(schema_lookup['concert_singer']['embeddings'].shape)
    print(schema_lookup['concert_singer']['schema'])
    print(schema_lookup['concert_singer'].keys())

    # search_column
    db = schema_lookup['concert_singer']
    result = search_column(
        query=['song name', 'singer age', 'song release year'],
        col_dicts=db['col_dicts'],
        embeddings=db['embeddings'],
        topk=3
    )
    print(result)

    # search_value: across the whole DB, then narrowed to one table/column
    db_id = 'activity_1'
    col_dicts = schema_lookup[db_id]['col_dicts']
    query = 'soccer activity'
    print(search_value(query, db_id, col_dicts, DB_BASE_PATH, table = None, column = None))
    print(search_value(query, db_id, col_dicts, DB_BASE_PATH, table = 'Activity', column = 'activity_name'))

    # graph check
    G = schema_lookup['concert_singer']['graph']
    print("Nodes:", G.number_of_nodes())  # should be 21
    print("Edges:", G.number_of_edges())

    # FindShortestPath
    list_of_paths = FindShortestPath(db_id = 'activity_1', schema_lookup= schema_lookup,
                        start=["Student.Fname", "Faculty.name"],
                        end=["Activity.activity_name"],
                        )
    print(list_of_paths)

    # ExecuteSQL
    result = ExecuteSQL(
        "SELECT Song_Name, Song_release_year FROM singer ORDER BY Age LIMIT 1",
        db_id="concert_singer",
        db_base_path=DB_BASE_PATH
    )
    print(result)
