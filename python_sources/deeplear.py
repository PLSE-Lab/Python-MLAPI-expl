import nltk 
import gensim
sample="""Renewed fighting has broken out in South Sudan between forces loyal to the president and vice-president. A reporter in the capital, Juba, told the BBC gunfire and large explosions could be heard all over the city; he said heavy artillery was being used. More than 200 people are reported to have died in clashes since Friday. The latest violence came hours after the UN Security Council called on the warring factions to immediately stop the fighting. In a unanimous statement, the council condemned the violence "in the strongest terms" and expressed "particular shock and outrage" at attacks on UN sites. It also called for additional peacekeepers to be sent to South Sudan.
Chinese media say two Chinese UN peacekeepers have now died in Juba. Several other peacekeepers have been injured, as well as a number of civilians who have been caught in crossfire. The latest round of violence erupted when troops loyal to President Salva Kiir and first Vice-President Riek Machar began shooting at each other in the streets of Juba. Relations between the two men have been fractious since South Sudan won independence from Sudan in 2011.
Their forces have fought a civil war. But despite a peace deal last year ending the conflict, both sides retain their military capabilities and have continued to accuse each other of bad faith. On Monday, there were reports of tanks on the streets of Juba and clashes close to the airport and UN camps sheltering civilians. The US embassy warned of "serious fighting" taking place. A BBC correspondent in the Kenyan capital, Nairobi, said it was not clear if Mr Kiir and Mr Machar remained in control of their forces. A UN spokeswoman in Juba, Shantal Persaud, said fighting over the past few days had caused hundreds of internally displaced people to take refuge in UN premises. She said both South Sudanese leaders were responsible for implementing last year's peace agreement, which included a permanent ceasefire and the deployment of forces away from Juba. Information Minister Michael Makuei told the BBC that the situation in the city was "under full control" and civilians who had fled should return to their homes. Mr Machar's military spokesman, Col William Gatjiath, accused officials loyal to the president of lying, and said there had been at least 10 hours of clashes on Sunday. "The situation in South Sudan is uncontrollable because Salva Kiir and his followers are not ready to follow the peace agreement," he said. 
"""

sentences = nltk.sent_tokenize(sample)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
model = gensim.models.Word2Vec(tokenized_sentences, min_count=1)

model.most_similar(positive=['Sudan'], topn=5)
model.most_similar(positive=['Sudan'], negative=['UN'], topn=5)
model.most_similar(positive=['Sudan', 'UN'], topn=5)

model.similarity('South', 'Sudan')
model.similarity('peace', 'Sudan')