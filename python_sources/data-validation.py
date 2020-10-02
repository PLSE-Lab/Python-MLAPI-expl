import cerberus
import json


# Using the packages mentioned in the “Tools for data validation”, check for the following things in your data:
# - Make sure that you can read the file in.
with open('../input/search', 'r') as f:
    document = json.loads(f.read())
    
    # - Check for missing data.
    # - If applicable: make sure you have the columns you expect for tabular data, or the keys you expect for hierarchical data.
    # - If applicable: check that the data types in the file you’ve read in are the ones you expect. (You might find it helpful to refer to the metadata here.)
    # - Optional: Validate another feature of your data that’s relevant to the thing you would use the data for, like character encoding.
    schema = {
        'response': {
            'type': 'dict',
            'required': True,
            'schema': {
                'numFound': {'type': 'integer', 'required': True, 'empty': False},
                'start': {'type': 'integer', 'required': True, 'empty': False},
                'maxScore': {'type': 'float', 'required': True, 'empty': False},
                'docs': {'type': 'list', 'required': True, 'schema': {
                    'type': 'dict', 'schema': {
                        'id': {'type': 'string', 'required': True, 'empty': False},
                        'journal': {'type': 'string', 'required': True, 'empty': False},
                        'eissn': {'type': 'string', 'required': True, 'empty': False},
                        'publication_date': {'type': 'string', 'required': True, 'empty': False},
                        'article_type': {'type': 'string', 'required': True, 'empty': False},
                        'author_display': {'type': 'list', 'schema': {'type': 'string'}, 'required': True},
                        'abstract': {'type': 'list', 'schema': {'type': 'string'}, 'required': True},
                        'title_display': {'type': 'string', 'required': True, 'empty': False},
                        'score': {'type': 'float', 'required': True, 'empty': False}
                    }
                }}
            }
        }
    }
    validator = cerberus.Validator(schema)
    result = validator.validate(document)
    print(result)
