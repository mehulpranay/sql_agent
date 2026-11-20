"""
exemplars.py  --  the two few-shot examples shown to the agent before every question.

Copied verbatim from the agent notebook (cell 8), only re-formatted:
    exemplar_1 : academic DB (empty tables) -> search_column, retry, FindShortestPath, ExecuteSQL
    exemplar_2 : activity_1 DB -> search_value, search_column, FindShortestPath, UNION query
"""

exemplar_1 = [{'role': 'user',
  'content': 'Database Schema:\n'
             'Tables: author, conference, domain, domain_author, domain_conference, journal, domain_journal, '
             'keyword, domain_keyword, publication, domain_publication, organization, publication_keyword, '
             'writes, cite\n'
             'ForeignKeys:\n'
             'domain_author.aid = author.aid\n'
             'domain_author.did = domain.did\n'
             'domain_conference.cid = conference.cid\n'
             'domain_conference.did = domain.did\n'
             'domain_journal.did = domain.did\n'
             'domain_journal.jid = journal.jid\n'
             'domain_keyword.did = domain.did\n'
             'domain_keyword.kid = keyword.kid\n'
             'publication.cid = conference.cid\n'
             'publication.jid = journal.jid\n'
             'domain_publication.did = domain.did\n'
             'domain_publication.pid = publication.pid\n'
             'publication_keyword.pid = publication.pid\n'
             'publication_keyword.kid = keyword.kid\n'
             'writes.aid = author.aid\n'
             'writes.pid = publication.pid\n'
             'cite.cited = publication.pid\n'
             'cite.citing = publication.pid\n'
             '\n'
             "Question: Return me the author in the 'University of Michigan' in Databases area whose papers "
             'have more than 5000 total citations.'},
 {'role': 'assistant',
  'content': 'Thought: This database has no data, so SearchValue would return empty results. I will proceed '
             'directly to finding relevant columns for: author name, university name, academic domain, and '
             'total citations.',
  'tool_calls': [{'id': 'call_001',
                  'type': 'function',
                  'function': {'name': 'search_column',
                               'arguments': '{"query": ["author name", "university name", "academic domain", '
                                            '"total citations"]}'}}]},
 {'role': 'tool',
  'content': "{'author name': [{'column': 'name', 'format': 'text', 'table': 'author', 'statistics': 'e.g: "
             "[]'}, {'column': 'name', 'format': 'text', 'table': 'journal', 'statistics': 'e.g: []'}, "
             "{'column': 'name', 'format': 'text', 'table': 'conference', 'statistics': 'e.g: []'}, "
             "{'column': 'name', 'format': 'text', 'table': 'keyword', 'statistics': 'e.g: []'}, {'column': "
             "'name', 'format': 'text', 'table': 'organization', 'statistics': 'e.g: []'}], 'university "
             "name': [{'column': 'name', 'format': 'text', 'table': 'journal', 'statistics': 'e.g: []'}, "
             "{'column': 'name', 'format': 'text', 'table': 'conference', 'statistics': 'e.g: []'}, "
             "{'column': 'name', 'format': 'text', 'table': 'author', 'statistics': 'e.g: []'}, {'column': "
             "'homepage', 'format': 'text', 'table': 'organization', 'statistics': 'e.g: []'}, {'column': "
             "'name', 'format': 'text', 'table': 'organization', 'statistics': 'e.g: []'}], 'academic "
             "domain': [{'column': 'did', 'format': 'number', 'table': 'domain_author', 'statistics': 'e.g: "
             "[]'}, {'column': 'did', 'format': 'number', 'table': 'domain_journal', 'statistics': 'e.g: "
             "[]'}, {'column': 'did', 'format': 'number', 'table': 'domain_conference', 'statistics': 'e.g: "
             "[]'}, {'column': 'name', 'format': 'text', 'table': 'domain', 'statistics': 'e.g: []'}, "
             "{'column': 'did', 'format': 'number', 'table': 'domain_keyword', 'statistics': 'e.g: []'}], "
             "'total citations': [{'column': 'cited', 'format': 'number', 'table': 'cite', 'statistics': "
             "'e.g: []'}, {'column': 'citation_num', 'format': 'number', 'table': 'publication', "
             "'statistics': 'e.g: []'}, {'column': 'citing', 'format': 'number', 'table': 'cite', "
             "'statistics': 'e.g: []'}]}",
  'tool_call_id': 'call_001'},
 {'role': 'assistant',
  'content': 'Thought: author.name, domain.name, and publication.citation_num are relevant. However '
             'university name was not found correctly — results show journal, conference, and author names '
             'but not a university/organization name column. A university is a type of organization, so I '
             'should search for organization name specifically.',
  'tool_calls': [{'id': 'call_002',
                  'type': 'function',
                  'function': {'name': 'search_column', 'arguments': '{"query": ["organization name"]}'}}]},
 {'role': 'tool',
  'content': "[{'column': 'name', 'format': 'text', 'table': 'organization', 'statistics': 'e.g: []'}, "
             "{'column': 'homepage', 'format': 'text', 'table': 'organization', 'statistics': 'e.g: []'}, "
             "{'column': 'name', 'format': 'text', 'table': 'journal', 'statistics': 'e.g: []'}, {'column': "
             "'name', 'format': 'text', 'table': 'conference', 'statistics': 'e.g: []'}, {'column': 'name', "
             "'format': 'text', 'table': 'domain', 'statistics': 'e.g: []'}]",
  'tool_call_id': 'call_002'},
 {'role': 'assistant',
  'content': 'Thought: organization.name is the university name column. Columns to select: author.name. '
             'Columns to filter: domain.name, organization.name, publication.citation_num. I need join paths '
             'between these.',
  'tool_calls': [{'id': 'call_003',
                  'type': 'function',
                  'function': {'name': 'FindShortestPath',
                               'arguments': '{"start": ["author.name"], "end": ["domain.name", '
                                            '"organization.name", "publication.citation_num"]}'}}]},
 {'role': 'tool',
  'content': "[('author.name', 'domain.name', 'author.name <-> author.aid = domain_author.aid <-> "
             "domain_author.did = domain.did <-> domain.name'), ('author.name', 'organization.name', "
             "'author.name <-> author.oid = organization.oid <-> organization.name'), ('author.name', "
             "'publication.citation_num', 'author.name <-> author.aid = writes.aid <-> writes.pid = "
             "publication.pid <-> publication.citation_num')]",
  'tool_call_id': 'call_003'},
 {'role': 'assistant',
  'content': 'Thought: I have all join paths. The question asks for authors with more than 5000 total '
             'citations — this requires GROUP BY author.name with HAVING SUM(citation_num) > 5000.',
  'tool_calls': [{'id': 'call_004',
                  'type': 'function',
                  'function': {'name': 'ExecuteSQL',
                               'arguments': '{"sql": "SELECT author.name FROM author JOIN domain_author ON '
                                            'author.aid = domain_author.aid JOIN domain ON domain_author.did '
                                            '= domain.did JOIN organization ON author.oid = organization.oid '
                                            'JOIN writes ON author.aid = writes.aid JOIN publication ON '
                                            "writes.pid = publication.pid WHERE domain.name = 'Databases' "
                                            "AND organization.name = 'University of Michigan' GROUP BY "
                                            'author.name HAVING SUM(publication.citation_num) > 5000"}'}}]},
 {'role': 'tool', 'content': '[]', 'tool_call_id': 'call_004'},
 {'role': 'assistant',
  'content': 'Thought: The query returned empty results — this is expected as the database tables are empty. '
             'The SQL structure is correct and complete.',
  'tool_calls': [{'id': 'call_005', 'type': 'function', 'function': {'name': 'Done', 'arguments': '{}'}}]},
 {'role': 'tool', 'content': 'Task Complete', 'tool_call_id': 'call_005'}]

exemplar_2 = [{'role': 'user',
  'content': 'Database Schema:\n'
             'Tables: Student, Faculty, Activity, Participates_in, Faculty_Participates_in\n'
             'ForeignKeys:\n'
             'Participates_in.StuID = Student.stuid\n'
             'Participates_in.actid = Activity.actid\n'
             'Faculty_Participates_in.FacID = Faculty.FacID\n'
             'Faculty_Participates_in.actid = Activity.actid\n'
             '\n'
             'Question: Which female students and male professors participated in the Soccer activity? List '
             'their names.'},
 {'role': 'assistant',
  'content': "Thought: The question mentions 'Soccer activity' which is likely a specific cell value. I also "
             'need to find columns for female student, male professor, and name.',
  'tool_calls': [{'id': 'call_001',
                  'type': 'function',
                  'function': {'name': 'search_value',
                               'arguments': '{"query": "Soccer activity", "table": null, "column": null}'}}]},
 {'role': 'tool',
  'content': "[{'contents': 'Soccer', 'table': 'Activity', 'column': 'activity_name'}]",
  'tool_call_id': 'call_001'},
 {'role': 'assistant',
  'content': "Thought: Found 'Soccer' in Activity.activity_name. Now I need to find columns for female "
             'student, male professor, and name.',
  'tool_calls': [{'id': 'call_002',
                  'type': 'function',
                  'function': {'name': 'search_column',
                               'arguments': '{"query": ["female student", "male professor", "name"]}'}}]},
 {'role': 'tool',
  'content': "{'female student': [{'column': 'Sex', 'format': 'text', 'table': 'Student', 'statistics': "
             '"e.g: [\'F\', \'M\']"}, {\'column\': \'Sex\', \'format\': \'text\', \'table\': \'Faculty\', '
             '\'statistics\': "e.g: [\'M\', \'F\']"}, {\'column\': \'Rank\', \'format\': \'text\', '
             '\'table\': \'Faculty\', \'statistics\': "e.g: [\'Instructor\', \'Professor\', '
             '\'AsstProf\']"}], \'male professor\': [{\'column\': \'Sex\', \'format\': \'text\', \'table\': '
             '\'Faculty\', \'statistics\': "e.g: [\'M\', \'F\']"}, {\'column\': \'Sex\', \'format\': '
             '\'text\', \'table\': \'Student\', \'statistics\': "e.g: [\'F\', \'M\']"}, {\'column\': '
             '\'Rank\', \'format\': \'text\', \'table\': \'Faculty\', \'statistics\': "e.g: [\'Instructor\', '
             '\'Professor\', \'AsstProf\']"}], \'name\': [{\'column\': \'activity_name\', \'format\': '
             '\'text\', \'table\': \'Activity\', \'statistics\': "e.g: [\'Soccer\', \'Baseball\', '
             '\'Football\']"}, {\'column\': \'Lname\', \'format\': \'text\', \'table\': \'Faculty\', '
             '\'statistics\': "e.g: [\'Kumar\', \'Goldman\']"}, {\'column\': \'LName\', \'format\': '
             '\'text\', \'table\': \'Student\', \'statistics\': "e.g: [\'Smith\', \'Kim\']"}]}',
  'tool_call_id': 'call_002'},
 {'role': 'assistant',
  'content': 'Thought: Student.Sex identifies female students. Faculty.Sex and Faculty.Rank identify male '
             "professors. However 'name' returned activity_name and only last name columns — I need both "
             'first name and last name for both Student and Faculty. Let me search more specifically.',
  'tool_calls': [{'id': 'call_003',
                  'type': 'function',
                  'function': {'name': 'search_column',
                               'arguments': '{"query": ["first name", "last name"]}'}}]},
 {'role': 'tool',
  'content': "{'first name': [{'column': 'Fname', 'format': 'text', 'table': 'Faculty', 'statistics': "
             '"e.g: [\'Baruch\', \'Paul\', \'Brian\']"}, {\'column\': \'Fname\', \'format\': \'text\', '
             '\'table\': \'Student\', \'statistics\': "e.g: [\'Linda\', \'Tracy\', \'Shiela\']"}, '
             "{'column': 'activity_name', 'format': 'text', 'table': 'Activity', 'statistics': "
             '"e.g: [\'Soccer\', \'Baseball\']"}], \'last name\': [{\'column\': \'Lname\', \'format\': '
             '\'text\', \'table\': \'Faculty\', \'statistics\': "e.g: [\'Kumar\', \'Goldman\', '
             '\'Cauwenberghs\']"}, {\'column\': \'LName\', \'format\': \'text\', \'table\': \'Student\', '
             '\'statistics\': "e.g: [\'Smith\', \'Kim\', \'Jones\']"}]}',
  'tool_call_id': 'call_003'},
 {'role': 'assistant',
  'content': 'Thought: Columns to select: Student.Fname, Student.LName, Faculty.Fname, Faculty.Lname. Filter '
             'column: Activity.activity_name. The question asks for two separate groups (female students AND '
             'male professors) so I will need a UNION query. Let me find join paths for both groups.',
  'tool_calls': [{'id': 'call_004',
                  'type': 'function',
                  'function': {'name': 'FindShortestPath',
                               'arguments': '{"start": ["Student.Fname", "Faculty.Fname"], "end": '
                                            '["Activity.activity_name"]}'}}]},
 {'role': 'tool',
  'content': "[('Student.Fname', 'Activity.activity_name', 'Student.Fname <-> Student.stuid = "
             "Participates_in.StuID <-> Participates_in.actid = Activity.actid <-> Activity.activity_name'), "
             "('Faculty.Fname', 'Activity.activity_name', 'Faculty.Fname <-> Faculty.FacID = "
             'Faculty_Participates_in.FacID <-> Faculty_Participates_in.actid = Activity.actid <-> '
             "Activity.activity_name')]",
  'tool_call_id': 'call_004'},
 {'role': 'assistant',
  'content': 'Thought: I have both join paths. The question asks for female students AND male professors as '
             'two separate groups — this requires a UNION query, not a single JOIN.',
  'tool_calls': [{'id': 'call_005',
                  'type': 'function',
                  'function': {'name': 'ExecuteSQL',
                               'arguments': '{"sql": "SELECT Student.Fname, Student.LName FROM Student JOIN '
                                            'Participates_in ON Student.stuid = Participates_in.StuID JOIN '
                                            'Activity ON Participates_in.actid = Activity.actid WHERE '
                                            "Activity.activity_name = 'Soccer' AND Student.Sex = 'F' UNION "
                                            'SELECT Faculty.Fname, Faculty.Lname FROM Faculty JOIN '
                                            'Faculty_Participates_in ON Faculty.FacID = '
                                            'Faculty_Participates_in.FacID JOIN Activity ON '
                                            'Faculty_Participates_in.actid = Activity.actid WHERE '
                                            "Activity.activity_name = 'Soccer' AND Faculty.Sex = 'M' AND "
                                            'Faculty.Rank = \'Professor\'"}'}}]},
 {'role': 'tool',
  'content': "[('Linda', 'Smith'), ('Sarah', 'Schmidt'), ('Stacy', 'Prater'), ('Michael', 'Goodrich')]",
  'tool_call_id': 'call_005'},
 {'role': 'assistant',
  'content': 'Thought: The UNION query executed successfully returning female students and male professors '
             'who participated in Soccer.',
  'tool_calls': [{'id': 'call_006', 'type': 'function', 'function': {'name': 'Done', 'arguments': '{}'}}]},
 {'role': 'tool', 'content': 'Task Complete', 'tool_call_id': 'call_006'}]

