"""
demo_dbs.py  --  the databases visitors may query, with what a visitor needs to know about each.

A visitor only types a question, so they must already know what is in the database. The
description and example questions are that "data dictionary" for the demo.

Every example question below is one that appears in your runs/ logs. Edit the text freely, and add
more examples once you have tried them.
"""

DEMO_DATABASES = {
    'concert_singer': {
        'title': 'Concerts and singers',
        'description': 'Singers and their songs, concerts, and the stadiums where concerts are held.',
        'example_questions': [
            'Show the name and the release year of the song by the youngest singer.',
        ],
    },
    'student_transcripts_tracking': {
        'title': 'University students and degree programs',
        'description': ('Students, their addresses, the degree programs and courses they enrol in, '
                        'and the semesters and transcripts that record it.'),
        'example_questions': [
            'Find the last name of the students who currently live in the state of North Carolina '
            'but have not registered in any degree program.',
            'Who are enrolled in 2 degree programs in one semester? List the first name, middle '
            'name and last name and the id.',
        ],
    },
}
