from utils_for_automata import * 
import itertools 

import re

import py2neo
from py2neo import Node, Relationship

params=config(section='neo4j')
graph = py2neo.Graph(params['neo4j_conn_url'], auth=(params['neo4j_user'], params['neo4j_pass']))

num_sents, num_ents = 0, 0
session = Session(engine)
for sdoc in session.query(Sentences).yield_per(400).enable_eagerloads(False):
    if num_sents % 100 == 0:
        print("... {:d} sentences read, {:d} entities written"
                .format(num_sents, num_ents))
    matched_ents = find_matches(str(sdoc.sentence_tokenised), A)
    if len(matched_ents)<1:
        print("Error ", sdoc.sentence_tokenised)
    else:
        for pair in itertools.combinations(matched_ents, 2):
            tx = graph.begin()
            canonical_name=re.sub('[^A-Za-z0-9]+', '_', str(pair[0][1]))
            entity_id=str(pair[0][0])
            source_entity=Node("ENTITY",name=canonical_name, entity_id=entity_id, id=entity_id,ename=canonical_name)
            tx.create(source_entity)
            canonical_name=re.sub('[^A-Za-z0-9]+', '_', str(pair[1][1]))
            destination_entity=Node("ENTITY",name=canonical_name, entity_id=str(pair[1][0]),id=entity_id,ename=canonical_name)
            tx.create(destination_entity)
            sentence_key="{:s}:{:d}:{:d}".format(sdoc.article_id, sdoc.paragraph_id,sdoc.sentence_id)
            relation_source_destination=Relationship(source_entity,"REL",destination_entity,skey=sentence_key)
            tx.create(relation_source_destination)
            tx.commit()
            num_ents += 1
    num_sents += 1
session.close()
print("Complete")