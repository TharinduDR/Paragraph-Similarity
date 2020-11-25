import json

from paragraph_similarity.algo.doc2vec.algo import Doc2VecSimilarityModel
import logging

from paragraph_similarity.common.paragraph import Paragraph

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

doc2vec_model = Doc2VecSimilarityModel(vector_size=50, min_count=2, epochs=40)
doc2vec_model.training("book.txt")

incoming_paragraph = Paragraph("EU law traditionally is one of the core subjects for a qualifying law degree in "
                               "England & Wales." "It filters through to the ‘domestic’ law of member states as "
                               "well as some states that have bilateral " "Treaties with the EU, such as Norway, "
                               "which are bound to follow large chunks of EU law.")

memory_paragraphs = list()

memory_paragraph_1 = Paragraph("EU law traditionally is one of the core subjects for a qualifying law degree in "
                               "England & Wales." "It filters through to the ‘domestic’ law of member states as "
                               "well as some states that have bilateral " "Treaties with the EU, such as Norway, "
                               "which are bound to follow large chunks of EU law.")
memory_paragraph_2 = Paragraph("EU law traditionally is one of the core subjects for a qualifying law degree in "
                               "England & Wales." "It filters through to the ‘domestic’ law of member states as "
                               "well as some states that have bilateral " "Treaties with the EU, such as Norway, "
                               "which are bound to follow large chunks of EU law.")
memory_paragraph_3 = Paragraph("EU law traditionally is one of the core subjects for a qualifying law degree in "
                               "England & Wales." "It filters through to the ‘domestic’ law of member states as "
                               "well as some states that have bilateral " "Treaties with the EU, such as Norway, "
                               "which are bound to follow large chunks of EU law.")
memory_paragraph_4 = Paragraph("EU law traditionally is one of the core subjects for a qualifying law degree in "
                               "England & Wales." "It filters through to the ‘domestic’ law of member states as "
                               "well as some states that have bilateral " "Treaties with the EU, such as Norway, "
                               "which are bound to follow large chunks of EU law.")

memory_paragraphs.append(memory_paragraph_1)
memory_paragraphs.append(memory_paragraph_2)
memory_paragraphs.append(memory_paragraph_3)
memory_paragraphs.append(memory_paragraph_4)

results = doc2vec_model.calculate_most_similar(incoming_paragraph, memory_paragraphs, 2)
for result in results:
    print(json.dumps(result.__dict__))

