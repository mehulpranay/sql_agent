"""
agent_demo.py  --  ReAct NL2SQL agent on DeepSeek, built to show one thing:

    deterministic path tracing  +  executing the SQL  +  error feedback
    => the agent ends up with the correct SQL.

The LLM only decides WHAT to look for. The tools are deterministic:
    search_column / search_value  -> find the right columns and the exact stored values
    FindShortestPath              -> join path from the column graph (no guessing joins)
    ExecuteSQL                    -> run the query; a failure comes back as "SQL Error: ..."
                                     and the model fixes it on the next turn

Each run is checked by execution match: the predicted SQL and the gold SQL must return the same rows.

Needs:
    DEEPSEEK_API_KEY   in your environment (optional DEEPSEEK_MODEL, default 'deepseek-flash')
    DEEPSEEK_STRICT    default on: strict tool schemas via the beta endpoint (https://api.deepseek.com/beta).
                       Set DEEPSEEK_STRICT=0 to fall back to the stable endpoint with plain tool schemas.
    spider_schema_lookup.pkl from preprocessing.py   (SCHEMA_PKL_PATH)
    Spider's database folder                         (SPIDER_DB_PATH)
    Spider's dev set: dev.json next to the database folder (or SPIDER_DEV_JSON), else the HF "spider" dataset

Usage:
    python agent_demo.py                      # search the dev set for 2 runs that show the full story
    python agent_demo.py --showcase 1 --max-tries 60
    python agent_demo.py --index 0 5 17       # run specific dev-set questions and print each trace
"""

import os
import copy
import json
import pickle
import random
import argparse

from openai import OpenAI

from runtime_tools import search_value, search_column, FindShortestPath, ExecuteSQL
from runtime_tools import DB_BASE_PATH, SCHEMA_PKL_PATH
from exemplars import exemplar_1, exemplar_2

# Strict tool calling is a DeepSeek Beta feature: it needs the /beta base URL and "strict": true on every tool.
STRICT = os.environ.get('DEEPSEEK_STRICT', '1') != '0'
BASE_URL = 'https://api.deepseek.com/beta' if STRICT else 'https://api.deepseek.com'
MODEL = os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash')
MAX_TURNS = 12


# ----------------------------------------------------------------------------
# prompt + tool definitions
# ----------------------------------------------------------------------------

# Same prompt as the notebook; the only edit is the tool names (SearchValue -> search_value, ...)
# so they match the real tool names.
SYSTEM_PROMPT = """When presented with a question, you need to use specific tools to interact with a locally deployed SQLite database and craft an SQL query to retrieve the answer.

General Process:

1. If a question includes a specific string value or entity name, utilize the search_value tool to identify the corresponding cell value, column name, and table name.
2. Use search_column to locate the column that is semantically related to the query.
3. Determine the shortest path between the selected columns and the condition columns using FindShortestPath
4. Execute the SQL query using ExecuteSQL. The result from the final call to ExecuteSQL will be taken as the answer, so it is critical to ensure that the SQL query is accurately formulated and complete.

Crucial Patterns:
- You need to use your intelligence and reasoning to determine how columns in the database reflect the semantics of the question, then use search_column() to locate them. For instance, "the city Bishopville, SC" may be represented by two columns: T.city = 'Bishopville' AND T.state = 'SC'.
- Be aware of dirty data in the database. Use 'IS NOT NULL' to filter out missing values.- MUST construct SQL queries strictly according to the observed values. For example, 'Akiak ' includes a trailing space.
- DO NOT select irrelevant columns. (e.g. for the question "who has the max value", you should NOT return the max value itself, but if the question is "what is the max value, list the people", you should return the max value and the corresponding people.)
- Use numbers from 0 to 100 to represent percentages. For example, use *100 in SQL if necessary.
- DO NOT concatenate column in select statement. For example, `SELECT c1 || ' ' || c2` is not allowed. You should SELECT c1, c2 instead.

Remember:
- DO NOT fabricate column names. Column names must be based on actual observation.
- Call only ONE tool at a time per turn.
- Always generate a Thought before calling a tool.
- STOP generating after calling a tool — wait for the observation.
- Only call Done after ExecuteSQL has returned the final result rows.
- Please keep interactions concise and accurate, avoiding unnecessary details.
"""

# Same tool definitions as the notebook, with two changes for DeepSeek's strict mode:
#   - strict mode only supports object/string/number/integer/boolean/array/enum/anyOf, and "null" is not
#     in that list, so search_value's table/column are plain strings and "" means "search everywhere"
#     (the exemplars are converted the same way in adapt_exemplar below)
#   - "strict": true is added to every tool right after this list
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_column",
            "description": "Search for relevant columns in the database based on semantic meaning. Use this to find which columns are related to concepts in the question. Returns up to 5 most relevant columns per query term, and corresponding tables, data types, and sample statistics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of semantic descriptions of columns you are looking for. Even for a single search term, wrap it in a list. e.g. ['singer name'] or ['singer name', 'song release year']"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_value",
            "description": "This tool searches for a value in the database based on the query parameter. It utilizes SQLite LIKE query to find the most relevant cell value, and returns the actual value, the corresponding column name, and the table name. You can also specify the table and column parameters to search within a specific table or column. If these parameters are not specified, the tool will search across all tables and columns. Note: This tool does not index the id column because the id column is not a text field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The text snippet or value to look up in the database. Examples: 'John Doe', 'California'."
                    },
                    "table": {
                        "type": "string",
                        "description": "The name of the specific table to restrict the search to. If you want to search all tables, you must explicitly pass an empty string \"\"."
                    },
                    "column": {
                        "type": "string",
                        "description": "The name of the specific column to restrict the search to. If you want to search all columns, you must explicitly pass an empty string \"\"."
                    }
                },
                "required": ["query", "table", "column"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "FindShortestPath",
            "description": "This function identifies the shortest path between two sets of columns in a database schema, treated as an undirected linked graph. In this graph, nodes represent columns, one type of edges represent foreign key constraints (linking tables) and another type of edge represent all the columns in a table joined to each other within the table (for in-table traversal) . The parameters start and end accept either a single column or a list of columns specified in the {table}.{column} format. The function returns the shortest path between each possible pair of start and end nodes, which can be utilized to construct SQL queries. In the returned path, <-> indicates traversal within the same table, and = indicates a foreign key join condition between two tables. Use the = conditions directly as JOIN ON clauses in your SQL query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of starting column nodes specified in the {table}.{column} format. Even for a single column, wrap it in a list. Examples: ['users.id'] or ['users.id', 'orders.user_id']"
                    },
                    "end": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of destination column nodes specified in the {table}.{column} format. Even for a single column, wrap it in a list. Examples: ['products.id'] or ['products.id', 'categories.id']"
                    }
                },
                "required": ["start", "end"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ExecuteSQL",
            "description": "Executes a SQL query against the database. Returns the result rows if successful, or an error message if the query fails. The result of the final ExecuteSQL call will be taken as the answer — ensure the query is complete and accurate before calling this.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The complete, valid SQL query to execute against the database. Example: 'SELECT name FROM users WHERE id = 1;'"
                    }
                },
                "required": ["sql"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Done",
            "description": "Call this when you have executed the final SQL query using ExecuteSQL and have the answer. Only call this after ExecuteSQL has returned the result rows.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
]


if STRICT:
    for _tool in TOOLS:
        _tool['function']['strict'] = True


def adapt_exemplar(messages):
    """
    The exemplars call search_value with null for "search everywhere"; the tool schema above
    asks for "" instead. Convert their search_value arguments so the examples match the schema.
    """
    adapted = copy.deepcopy(messages)
    for m in adapted:
        for tc in m.get('tool_calls', []):
            if tc['function']['name'] == 'search_value':
                a = json.loads(tc['function']['arguments'])
                a = {k: ('' if v is None else v) for k, v in a.items()}
                tc['function']['arguments'] = json.dumps(a)
    return adapted


def build_initial_message(question, db_id, schema_lookup):

    schema_str = schema_lookup[db_id]['schema']
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    messages.extend(adapt_exemplar(exemplar_2))
    messages.extend(adapt_exemplar(exemplar_1))

    # same layout as the user turns inside the exemplars
    messages.append({'role': 'user',
                     'content': f'Database Schema:\n{schema_str}\n\nQuestion: {question}'})
    return messages


def make_client():
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise SystemExit('DEEPSEEK_API_KEY is not set in the environment.')
    return OpenAI(api_key=api_key, base_url=BASE_URL)


# ----------------------------------------------------------------------------
# the ReAct loop
# ----------------------------------------------------------------------------

def run_agent(client, question, db_id, schema_lookup, db_base_path):
    """
    Runs the ReAct loop for one question.
    Returns a dict with the predicted SQL, a step-by-step trace and bookkeeping flags.
    """

    messages = build_initial_message(question, db_id, schema_lookup)
    col_dicts = schema_lookup[db_id]['col_dicts']
    embeddings = schema_lookup[db_id]['embeddings']

    trace = []
    last_sql = None
    sql_status = 'SQL never generated'
    turn = 0

    while turn < MAX_TURNS:

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.0,
            extra_body={"thinking": {"type": "disabled"}},   # thinking mode is on by default
        )

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            break

        tool_call = message.tool_calls[0]       # one tool per turn
        tool_name = tool_call.function.name
        tool_id = tool_call.id

        step = {'turn': turn, 'thought': message.content, 'tool': tool_name,
                'args': None, 'observation': None, 'sql_status': None}
        finished = False

        try:
            args = json.loads(tool_call.function.arguments or '{}')
            step['args'] = args

            if tool_name == 'search_value':
                observation = search_value(query=args['query'],
                                           db_id=db_id,
                                           col_dicts=col_dicts,
                                           db_base_path=db_base_path,
                                           table=args.get('table') or None,
                                           column=args.get('column') or None)

            elif tool_name == 'search_column':
                query = args['query']
                if isinstance(query, str):      # guard: the tool expects a list
                    query = [query]
                observation = search_column(query=query,
                                            col_dicts=col_dicts,
                                            embeddings=embeddings)

            elif tool_name == 'FindShortestPath':
                observation = FindShortestPath(db_id=db_id,
                                               schema_lookup=schema_lookup,
                                               start=args['start'],
                                               end=args['end'])

            elif tool_name == 'ExecuteSQL':
                last_sql = args['sql']
                sql_status, rows = ExecuteSQL(sql=last_sql, db_id=db_id, db_base_path=db_base_path)
                step['sql_status'] = sql_status

                if sql_status != 'Successful':
                    observation = f'SQL Error: {rows}'              # error feedback to the model
                elif len(rows) > 10:
                    observation = f'{rows[:10]}....{len(rows)} total rows, showing first 10'
                else:
                    observation = rows

            elif tool_name == 'Done':
                observation = 'Task Complete'
                finished = True

            else:
                observation = f'Unknown tool name: {tool_name}'

        except Exception as e:
            observation = f'Tool call failed: {e}'

        step['observation'] = str(observation)
        trace.append(step)

        messages.append({'role': 'tool', 'content': str(observation), 'tool_call_id': tool_id})

        # every tool_call_id needs an answer, even if the model ignored "one tool at a time"
        for extra in message.tool_calls[1:]:
            messages.append({'role': 'tool',
                             'content': 'Skipped: call only one tool per turn.',
                             'tool_call_id': extra.id})

        turn += 1
        if finished:
            break

    return {
        'db_id': db_id,
        'question': question,
        'predicted_sql': last_sql,
        'sql_status': sql_status,
        'turns_used': turn,
        'hit_max_turns': turn == MAX_TURNS,
        'trace': trace,
        'conversation': messages,
    }


# ----------------------------------------------------------------------------
# checking the answer
# ----------------------------------------------------------------------------

def execution_match(pred_sql, gold_sql, db_id, db_base_path):
    """True if predicted and gold SQL return the same rows (order only matters if gold has ORDER BY)."""

    if not pred_sql:
        return False

    pred_status, pred_rows = ExecuteSQL(pred_sql, db_id, db_base_path)
    gold_status, gold_rows = ExecuteSQL(gold_sql, db_id, db_base_path)

    if pred_status != 'Successful' or gold_status != 'Successful':
        return False

    if 'order by' in gold_sql.lower():
        return pred_rows == gold_rows

    return sorted(map(repr, pred_rows)) == sorted(map(repr, gold_rows))


def describe_run(run):
    """Flags that say what the run actually showed."""

    used_path = any(s['tool'] == 'FindShortestPath' for s in run['trace'])

    sql_steps = [s['sql_status'] for s in run['trace'] if s['tool'] == 'ExecuteSQL']
    recovered = 'Unsuccessful' in sql_steps and 'Successful' in sql_steps[sql_steps.index('Unsuccessful'):]

    return used_path, recovered


# ----------------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------------

def short(text, limit=700):
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f' ...({len(text) - limit} more chars)'


def print_run(run, gold_sql, correct):

    bar = '=' * 70
    print(f'\n{bar}\nQUESTION [{run["db_id"]}]: {run["question"]}\n{bar}')

    for step in run['trace']:
        print(f'\n--- TURN {step["turn"]} ---')
        thought = (step['thought'] or '').strip()
        if thought.startswith('Thought:'):
            thought = thought[len('Thought:'):].strip()
        print(f'[Thought] {thought}')

        if step['args']:
            args = ', '.join(f'{k}={v}' for k, v in step['args'].items())
            print(f'[Action]  {step["tool"]}({args})')
        else:
            print(f'[Action]  {step["tool"]}()')

        if step['sql_status'] == 'Unsuccessful':
            print(f'[Result]  FAILED -> {short(step["observation"])}')
        else:
            print(f'[Result]  {short(step["observation"])}')

    used_path, recovered = describe_run(run)
    print(f'\n{bar}\nPredicted SQL : {run["predicted_sql"]}')
    print(f'Gold SQL      : {gold_sql}')
    print(f'Execution match with gold : {correct}')
    print(f'Turns used: {run["turns_used"]} | used FindShortestPath: {used_path} | recovered from an SQL error: {recovered}')
    print(bar)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def load_dev_examples():

    dev_json = os.environ.get(
        'SPIDER_DEV_JSON',
        os.path.join(os.path.dirname(os.path.normpath(DB_BASE_PATH)), 'dev.json'),
    )
    if os.path.isfile(dev_json):
        with open(dev_json, encoding='utf-8') as f:
            return json.load(f)

    from datasets import load_dataset
    return list(load_dataset('spider', split='validation'))


def run_and_check(client, example, schema_lookup):

    run = run_agent(client, example['question'], example['db_id'], schema_lookup, DB_BASE_PATH)
    correct = execution_match(run['predicted_sql'], example['query'], example['db_id'], DB_BASE_PATH)
    return run, correct


def main():

    parser = argparse.ArgumentParser(description='NL2SQL agent demo on DeepSeek')
    parser.add_argument('--index', type=int, nargs='*', help='run these dev-set indices and print each trace')
    parser.add_argument('--showcase', type=int, default=2, help='how many showcase runs to look for (default 2)')
    parser.add_argument('--max-tries', type=int, default=40, help='dev questions to try while looking (default 40)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    client = make_client()

    with open(SCHEMA_PKL_PATH, 'rb') as f:
        schema_lookup = pickle.load(f)

    examples = load_dev_examples()

    # specific questions
    if args.index:
        for i in args.index:
            example = examples[i]
            run, correct = run_and_check(client, example, schema_lookup)
            print_run(run, example['query'], correct)
        return

    # showcase: look for runs that are correct, used the graph path, and recovered from an SQL error
    order = random.Random(args.seed).sample(range(len(examples)), min(args.max_tries, len(examples)))

    correct_runs = []       # (run, gold_sql) that are correct and used FindShortestPath
    n_tried = n_correct = 0
    full_story = 0

    for i in order:
        example = examples[i]
        print(f'[trying dev #{i}] {example["db_id"]}: {example["question"][:70]}')

        try:
            run, correct = run_and_check(client, example, schema_lookup)
        except Exception as e:
            print(f'  API/run error: {e}')
            continue

        n_tried += 1
        n_correct += correct
        used_path, recovered = describe_run(run)

        if correct and used_path:
            correct_runs.append((run, example['query'], recovered))
            if recovered:
                full_story += 1
                if full_story >= args.showcase:
                    break

    print(f'\nTried {n_tried} questions, {n_correct} matched the gold result, '
          f'{full_story} showed path tracing + SQL error recovery + correct answer.')

    # error-recovery runs first, then the other correct runs that used the graph path
    correct_runs.sort(key=lambda item: not item[2])

    if not correct_runs:
        print('No correct run with path tracing yet; try a larger --max-tries.')

    for run, gold_sql, _ in correct_runs[:args.showcase]:
        print_run(run, gold_sql, True)


if __name__ == '__main__':
    main()