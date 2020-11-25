import abc

from paragraph_similarity.common.paragraph import Paragraph


class SimilarityModel:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def calculate_most_similar(self, input_text: Paragraph, memory_texts: list, n: int):
        """Method documentation"""
        return

