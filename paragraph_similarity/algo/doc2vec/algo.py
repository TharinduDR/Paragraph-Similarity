import gensim
import smart_open
from scipy import spatial
import numpy as np

from paragraph_similarity.common.paragraph import Paragraph
from paragraph_similarity.common.result import Result
from paragraph_similarity.common.similarity_model import SimilarityModel


class Doc2VecSimilarityModel(SimilarityModel):

    def __init__(self, vector_size, min_count, epochs):
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)

    @staticmethod
    def read_corpus(fname, tokens_only=False):
        with smart_open.open(fname, encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                tokens = gensim.utils.simple_preprocess(line)
                if tokens_only:
                    yield tokens
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    def training(self, train_file_name):
        train_corpus = self.read_corpus(train_file_name)
        self.model.build_vocab(train_corpus)
        self.model.train(train_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def inference(self, text_1: Paragraph, text_2: Paragraph):

        return spatial.distance.cosine(self.model.infer_vector(text_1.words),
                                       self.model.infer_vector(text_2.words))

    def calculate_most_similar(self, input_text: Paragraph, memory_paragraphs: list, n: int):
        similarity_list = []
        complete_result_list = []
        for memory_paragraph in  memory_paragraphs:
            similarity_list.append(self.inference(input_text, memory_paragraph))

        arr = np.array(similarity_list)
        filtered_arr = arr.argsort()[:n]

        for index in filtered_arr.tolist():
            complete_result_list.append(Result(memory_paragraphs[index].text, similarity_list[index]))

        return complete_result_list




