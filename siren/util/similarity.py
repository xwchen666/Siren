import abc
from abc import abstractmethod
import numpy as np
import Levenshtein as Lev
import re

class Similarity(abc.ABC):
    def preprocess(self, txt):
      """Preprocess text before WER caculation."""
    
      # Lowercase, remove \t and new line.
      txt = re.sub(r'[\t\n]', ' ', txt.lower())
    
      # Remove punctuation before space.
      txt = re.sub(r'[,.\?!]+ ', ' ', txt)
    
      # Remove punctuation before end.
      txt = re.sub(r'[,.\?!]+$', ' ', txt)
    
      # Remove punctuation after space.
      txt = re.sub(r' [,.\?!]+', ' ', txt)
    
      # Remove quotes, [, ], ( and ).
      txt = re.sub(r'["\(\)\[\]]', '', txt)
    
      # Remove extra space.
      txt = re.sub(' +', ' ', txt.strip())
    
      return txt

    def compute_similarity(self, t1, t2):
        """
        Compute the similarity of t1 and t2

        Parameters
        ----------
        t1: :class: `numpy.array` of str
            The first text
        t2: :class: `numpy.array` of str
            The second text

        Returns
        -------
        numpy.array of float 
        """
        raise NotImplementedError

    def __call__(self, t1, t2):
        return self.compute_similarity(t1, t2)

class ExactSimilarity(Similarity):
    def compute_similarity(self, t1, t2):
        if isinstance(t1, str):
            if isinstance(t2, str):
                return int(t1 == t2)
            else:
                return int(t1 == t2[0])
        t1 = np.array(t1)
        t2 = np.array(t2)
        res = t1 == t2
        return res.astype(np.float) 

class WordSimilarity(Similarity):
    def compute_similarity(self, t1, t2):
        if isinstance(t1, str):
            if isinstance(t2, str):
                return self.wer(t1, t2)
            else:
                return self.wer(t1, t2[0])
        sim = np.array([self.wer(s1, s2) for s1, s2 in zip(t1, t2)])
        return sim
    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Parameters
        ----------
        s1: str
            Hypothesis string
        s2: str
            Reference string

        Returns
        -------
        float
            Word error
        """
        # preprocess the txt
        s1 = self.preprocess(s1)
        s2 = self.preprocess(s2)
        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))
        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        nref = max(len(s2.split()), 1)
        return 1 - Lev.distance(''.join(w1), ''.join(w2)) / nref

class CharSimilarity(Similarity):   
    def compute_similarity(self, t1, t2):
        if isinstance(t1, str):
            if isinstance(t2, str):
                return self.cer(t1, t2)
            else:
                return self.cer(t1, t2[0])
        sim = np.array([self.cer(s1, s2) for s1, s2 in zip(t1, t2)])
        return sim

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.
        Parameter
        ---------
        s1: str 
            hypothesis str
        s2: str 
            ref string
        """
        # preprocess the txt
        s1 = self.preprocess(s1)
        s2 = self.preprocess(s2)
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return 1 - Lev.distance(s1, s2) / max(len(s2), 1)
