# NL2SQL agent: giving a text-to-SQL model tools

A ReAct agent for Spider that looks at the database before and while it writes SQL. This is Part 2 of the NL2SQL series. Part 1 fine-tuned Llama 3.1 8B with QLoRA and reached 85.59% execution accuracy on the Spider dev set: [Part 1 write-up](https://www.kaggle.com/code/mehulkumar99/blog-question-to-sql-query-spider-1).

## Claim

Many of the errors in the Part 1 failure taxonomy are grounding errors: a value spelled differently in the database than in the question, a join that produces hidden duplicates, a column or table that does not exist. An agent that can query the database can fix these, because the correction comes from the data and not from the model's guess. The agent does not fix interpretation errors, where the question or the gold query can be read in more than one way, and a prompt rule can introduce new errors.

The evidence is ten dev questions chosen from the fine-tuned model's failures. They show which kinds of errors the tools remove. They are not an accuracy figure. The full write-up, with one section per failure category, is here: [Part 2 write-up](https://docs.google.com/document/d/1AMv45QOjLP_pNrJvYFTXw5DdSenJ11ZhEOdjugVegX4/edit?usp=drivesdk&ouid=100398964835883099927).

## How the agent works

The agent runs a ReAct loop with a limit of 12 turns. Each turn the model writes a short thought and calls one tool; only the first tool call in a turn is executed. The prompt holds the schema (tables and foreign-key lines) and two worked examples. The model is DeepSeek, called at temperature 0 with thinking mode off and strict tool schemas.

| Tool | What it does | Deterministic part |
| --- | --- | --- |
| `search_column` | Takes short descriptions of the columns the question needs and returns the closest columns with table, type and three sample values. | MiniLM embeddings ranked by cosine similarity. |
| `search_value` | Takes a string from the question, optionally limited to a table and column, and returns stored values that match parts of the string, with their table and column. | Searches the database contents directly. |
| `FindShortestPath` | Takes start and end columns and returns the join conditions that connect them. | Shortest path over a graph built from foreign keys and same-table links. |
| `ExecuteSQL` | Runs the query on SQLite and returns rows or the error message. Results over 10 rows are truncated for the model. | The database engine. |
| `Done` | Ends the run and submits the last executed query. | None. |

The model decides which tool to call and writes the SQL. Every tool output comes from code, so the model reads facts about the database and does not have to recall them.

## Results on the ten questions

Each row links to the full log of the agent's run in [`runs/`](runs/).

| Index | Category | 85.59% model | Agent | Log |
| --- | --- | --- | --- | --- |
| 75 | 1, cardinality | Wrong | Correct in the logged run, no DISTINCT in a later run | [run_75](runs/run_75.txt) |
| 6 | 2, wrong column | Correct | Correct | [run_6](runs/run_6.txt) |
| 89 | 3, hallucination | Correct | Values match, alias differs from gold | [run_89](runs/run_89.txt) |
| 147 | 4, wrong table | Wrong | Correct | [run_147](runs/run_147.txt) |
| 439 | 5, aggregation semantics | Wrong | Correct in the latest run, wrong in an earlier run | [run_439](runs/run_439.txt) |
| 798 | 5, aggregation semantics | Wrong | Wrong (extra NULL filter from a prompt rule, aliases) | [run_798](runs/run_798.txt) |
| 744 | 6, set operations | Wrong | Correct | [run_744](runs/run_744.txt) |
| 113 | 7, case sensitivity | Wrong | Correct | [run_113](runs/run_113.txt) |
| 549 | Value spelling | Wrong | Correct | [run_549](runs/run_549.txt) |
| 533 | Loose gold query | Wrong | Wrong (gold ignores the semester) | [run_533](runs/run_533.txt) |

## Example: question 549

Question: find the last name of the students who currently live in the state of North Carolina but have not registered in any degree program.

| Turn | Action | Output | What it contributed |
| --- | --- | --- | --- |
| 0 | `search_column` for last name, state, degree program | Candidate columns | Real column names for all three concepts. |
| 1 | `FindShortestPath` from `Students.last_name` to the state and degree program columns | `Students.current_address_id = Addresses.address_id`, `Students.student_id = Student_Enrolment.student_id` | The join columns for three tables came from the foreign-key graph. |
| 2 | `ExecuteSQL` with `state = 'North Carolina'` | Empty result | The agent treated an empty result as a sign that the value might be wrong. |
| 3 | `search_value` for "North Carolina" | `'NorthCarolina'` | The database supplied the stored spelling. |
| 4 | `ExecuteSQL` with `state = 'NorthCarolina'` | Gleichner, Weimann | The corrected query returned rows and matches gold. |

The spelling cannot be inferred from the question or the schema. Full log: [run_549](runs/run_549.txt).

## Caveats

- The ten questions come from the fine-tuned model's failures. This shows which error types the tools remove and does not estimate accuracy on the full dev set.
- Each case is one saved run at temperature 0, and runs are not fully repeatable. In a later run the agent dropped DISTINCT in question 75, and it answered question 439 correctly after an earlier wrong run.
- The agent uses a much larger model than the 8B fine-tune, so model size and tools are not separated. A no-tools run of the same model has not been done.
- The 85.59% figure comes from a custom single-database evaluator that ignores row and column order but compares column names. The official Spider test-suite evaluator differs: it ignores names, respects ORDER BY and checks several database variants.
- The "Execution match" lines in the logs in `runs/` come from an older version of `agent_demo.py` and do not use `evaluator.py`.
- Some gold queries are loose, as in question 533.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
```

Download Spider 1.0 and point the scripts at its `database` folder:

```bash
export SPIDER_DB_PATH=/path/to/spider/database
export DEEPSEEK_API_KEY=...      # your DeepSeek key
```

Build the schema lookup once. It writes `spider_schema_lookup.pkl`:

```bash
python preprocessing.py
```

## Run

Run specific dev questions and print each trace:

```bash
python agent_demo.py --index 549
python agent_demo.py --index 75 147 744 2>&1 | tee runs/new_runs.txt
```

Search random dev questions for runs that were correct, used `FindShortestPath` and recovered from an SQL error:

```bash
python agent_demo.py --showcase 2
```

Optional environment variables: `DEEPSEEK_MODEL` (default `deepseek-flash`), `DEEPSEEK_STRICT=0` to use the stable endpoint without strict tool schemas, `SPIDER_DEV_JSON` for the path to `dev.json`, `SCHEMA_PKL_PATH` for the lookup file.

## Files

| File | Purpose |
| --- | --- |
| `preprocessing.py` | Builds the column-level schema lookup, embeddings and column graph. |
| `runtime_tools.py` | The four tools the agent calls. |
| `exemplars.py` | The two worked examples placed in the prompt. |
| `evaluator.py` | The execution-accuracy check from the Part 1 notebook, plus a values-only check. |
| `agent_demo.py` | The agent loop and the command-line runner. |
| `runs/` | Full logs of the ten runs used in the write-up. |