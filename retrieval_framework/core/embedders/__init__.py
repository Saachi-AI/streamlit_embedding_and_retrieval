"""
Embedders package for the retrieval framework.

Contains different embedding model implementations that convert text to vectors.
"""

from retrieval_framework.core.embedders.base import EmbedderInterface
from retrieval_framework.core.embedders.openai import OpenAIEmbedder
from retrieval_framework.core.embedders.cohere import CohereEmbedder

__all__ = [
    'EmbedderInterface',
    'OpenAIEmbedder',
    'CohereEmbedder'
] 