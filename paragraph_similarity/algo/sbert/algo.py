from paragraph_similarity.common.paragraph import Paragraph
from paragraph_similarity.common.result import Result
from paragraph_similarity.common.similarity_model import SimilarityModel
from sentence_transformers import SentenceTransformer, util
import torch


class SBERTSimilarityModel(SimilarityModel):
    def __init__(self):
        self.model = SentenceTransformer("paraphrase-MiniLM-L12-v2")

    def calculate_most_similar(self, input_text: Paragraph, memory_paragraphs: list, n: int):

        input_vector = self.model.encode(input_text.text, convert_to_tensor=True)
        texts = []
        for memory_paragraph in memory_paragraphs:
            temp_text = memory_paragraph.text
            texts.append(temp_text)
        memory_vectors = self.model.encode(texts, convert_to_tensor=True)

        cos_scores = util.pytorch_cos_sim(input_vector, memory_vectors)[0]
        top_results = torch.topk(cos_scores, k=n)

        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            format_score = score.double()
            results.append(Result(memory_paragraphs[idx].text, format_score.item()))

        return results