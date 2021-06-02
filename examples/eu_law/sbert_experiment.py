import json

from paragraph_similarity.algo.doc2vec.algo import Doc2VecSimilarityModel
import logging
import warnings


from paragraph_similarity.algo.sbert.algo import SBERTSimilarityModel
from paragraph_similarity.common.paragraph import Paragraph

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sbert_model = SBERTSimilarityModel()


incoming_paragraph = Paragraph("EU law traditionally is one of the core subjects for a qualifying law degree in "
                               "England & Wales." "It filters through to the ‘domestic’ law of member states as "
                               "well as some states that have bilateral " "Treaties with the EU, such as Norway, "
                               "which are bound to follow large chunks of EU law.")

memory_paragraphs = list()

memory_paragraph_1 = Paragraph("EU law traditionally is one of the core subjects for a qualifying law degree in "
                               "England & Wales." "It filters through to the ‘domestic’ law of member states as "
                               "well as some states that have bilateral " "Treaties with the EU, such as Norway, "
                               "which are bound to follow large chunks of EU law.")
memory_paragraph_2 = Paragraph("Functioning of the European Union (TFEU). When referred to generically in this course "
                               "the expression Treaties is used. These are framework Treaties. ")
memory_paragraph_3 = Paragraph("The relationship between the EEA, EFTA and EU member states is complex: The EU is a "
                               "member of the EEA. Each member state of the EU has membership of the EEA under that "
                               "auspice, but not in its own right.")


memory_paragraphs.append(memory_paragraph_1)
memory_paragraphs.append(memory_paragraph_2)
memory_paragraphs.append(memory_paragraph_3)
# memory_paragraphs.append(memory_paragraph_4)

results = sbert_model.calculate_most_similar(incoming_paragraph, memory_paragraphs, 2)
for result in results:
    print(json.dumps(result.__dict__))

