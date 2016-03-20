# -*- coding: utf-8 -*-
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from unicodedata import category
import re, sys, codecs
from scipy import stats
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import cPickle as pickle
import string

tokenizer = TreebankWordTokenizer()
stemmer = PorterStemmer()
word2num_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'thirteen':13, 'fourteen':14, 'fifteen':15, 'sixteen':16, 'seventeen':17, 'eighteen':18, 'nineteen':19, 'twenty':20, 'thirty':30, 'forty':40, 'fifty':50, 'sixty':60, 'seventy':70, 'eighty':80, 'ninety':90}

class StopWords:
   def __init__(self, fname='dat/misc/E.stoplist'):
      self.stopwords = set()
      with open(fname, 'r') as fin:
         for line in fin:
            self.stopwords.add(line.strip())

   def is_stopword(self, word):
      if word in self.stopwords:
         return True
      return False

stopwordsDict = StopWords()

def remove_non_ascii(s):
   return filter(lambda x: x in string.printable, s)

def mean(array):
   if len(array) == 0:
      return None
   return sum(array) / float(len(array))

def get_unigrams(array, pos=None):
   unigrams = array
   if pos:
      print unigrams, pos
      assert len(unigrams) == len(pos)
      unigrams_pos = ['%s_pos=%d' % (unigram, p) for unigram, p in zip(unigrams, pos)]
   else:
      unigrams_pos = []
   return unigrams, unigrams_pos

def get_bigrams(array, pos=None):
   if len(array) < 1:
      return [], []
   new_array = ['<S>'] + array + ['<E>']
   bigrams = ['%s^%s' % (new_array[i], new_array[i+1]) for i in range(len(new_array)-1)]
   if pos:
      new_pos = [1] + pos
      assert len(new_pos) == len(bigrams)
      bigrams_pos = ['%s_pos=%d' % (bigram, p) for bigram, p in zip(bigrams, pos)]
   else:
      bigrams_pos = []
   return bigrams, bigrams_pos

def get_trigrams(array, pos=None):
   if len(array) < 2:
      return [], []
   new_array = ['<SS>', '<S>'] + array + ['<E>', '<EE>']
   trigrams = ['%s^%s^%s' % (new_array[i], new_array[i+1], new_array[i+2]) for i in range(len(new_array)-2)]
   if pos:
      new_pos = [1, 1] + pos
      assert len(new_pos) == len(trigrams)
      trigrams_pos = ['%s_pos=%d' % (trigram, p) for trigram, p in zip(trigrams, pos)]
   else:
      trigrams_pos = []
   return trigrams, trigrams_pos

def bin_val(idx, N):
   # TODO: should not use N in incremental computation
   #pos = (idx+1) / float(N)
   #for i, l in enumerate([0, 0.2, 0.4, 0.8, 1]):
   #   if pos <= l:
   #      return i
   if idx < 5:
      return 1
   elif idx < 10:
      return 2
   elif idx < 15:
      return 3
   else:
      return 4

def is_stopword(word):
   return word[0] == '.' or stopwordsDict.is_stopword(word)

def is_sent_end(text):
   """
   return end-of-sent punctuation
   """
   m = re.search(ur'([ ]*[\.\?!][ ]*["]*)$', text, re.UNICODE)
   if m:
      return m.group(0)
   else:
      return None

def dump(obj, filename):
   with open(filename, 'wb') as fout:
      pickle.dump(obj, fout)

def load(filename):
   with open(filename, 'rb') as fin:
      obj = pickle.load(fin)
   return obj

def print_table(row, col, dat):
   row_format = '{:>15}' * (len(dat[0])+1)
   print row_format.format("", *row)
   for i, row in enumerate(dat):
      print row_format.format(col[i], *row)

def plot_histogram(data, filename, nbins=50):
   hist, bins = np.histogram(data, bins=nbins)
   width = 0.7 * (bins[1] - bins[0])
   center = (bins[:-1] + bins[1:]) / 2
   fig, ax = plt.subplots()
   ax.bar(center, hist, align='center', width=width)
   fig.savefig(filename)
   return hist, bins

def which_bin(x, bins):
   for i, b in enumerate(bins):
      if x < b:
         return i - 1
   return len(bins) - 2

def plot_stacked_histogram(ref_data, data, filename, nbins=50, mask=None):
   '''
   ref_data and data entries must be correspondent to each other
   '''
   if mask:
      new_ref_data = []
      new_data = []
      for m, rd, d in zip(mask, ref_data, data):
         if m:
            new_ref_data.append(rd)
            new_data.append(d)
      ref_data = new_ref_data
      data = new_data

   hist, bins = np.histogram(ref_data, bins=nbins)
   width = 0.7 * (bins[1] - bins[0])
   center = (bins[:-1] + bins[1:]) / 2
   fig, ax = plt.subplots()
   #tmp = hist[0]
   #hist[0] = 0
   ax.bar(center, hist, align='center', width=width, color='b')
   #hist[0] = tmp

   hist2 = [0] * len(hist)
   for rd, d in zip(ref_data, data):
      hist2[which_bin(rd, bins)] += d
   # average
   for i in range(len(hist2)):
      if hist[i] == 0:
         assert hist2[i] == 0
      else:
         hist2[i] /= float(hist[i])
   #tmp = hist2[0]
   #hist2[0] = 0
   ax.bar(center, hist2, align='center', width=width, color='r')
   #hist2[0] = tmp
   fig.savefig(filename)

   return hist, hist2, bins

def ttest(list1, list2):
   a1 = np.array(list1)
   a2 = np.array(list2)
   diff = a1 - a2
   t, prob = stats.ttest_rel(a1, a2)
   print np.mean(diff), np.std(diff), t, prob
   return np.mean(diff), np.std(diff), t, prob

#def ttest(arr1, arr2):
#   T, pvalue = stats.ttest_rel(arr1, arr2)
#   return T, pvalue

def ftest(arr1, arr2):
   std1 = np.std(arr1)
   std2 = np.std(arr2)
   F = std1 / std2
   df1 = len(std1) - 1
   df2 = len(std2) - 1
   pvalue = stats.f.cdf(F, df1, df2)
   return F, pvalue

def word2num(word):
   if '-' in word:
      ss = word.split('-')
      if len(ss) == 2 and ss[0] in word2num_dict and ss[1] in word2num_dict:
         return str(word2num_dict[ss[0]] + word2num_dict[ss[1]])
      return word
   else:
      if word in word2num_dict:
         return str(word2num_dict[word])
      return word

def is_punct(w):
   if re.match(r'[,"`\-\.\?!\'\(\)]+', w) or (isinstance(w, unicode) and len(w) == 1 and ((category(w).startswith('P') or category(w).startswith('S')))):
      return True
   return False

def remove_punct(sent):
   sent_nopunct = []
   for w in sent:
      cat = category(w)
      if w == u'-' or w == u"'" or not ((cat.startswith('P') or cat.startswith('S'))):
         sent_nopunct.append(w)
   return ''.join(sent_nopunct)

def tokenize(sent):
   return tokenizer.tokenize(sent)

def stem(toks):
   return [stemmer.stem(tok) for tok in toks]

def clean_tags(s):
   #print 'input:', s
   while True:
      s = ' '.join(s.split())
      match = re.findall(r'\([^\)\(]*\)|\([^\)\(]*$', s)
      if not match:
         break
      for m in match:
         key = m.split()[0][1:]
         start = len(key) + 2  # content start
         if m[-1] == ')':
            end = len(m) - 1
         else:
            end = len(m)
         ss = m[start:end]
         #print u'match: %s' % m, u'key: %s' % key, u'content: %s' % ss
         # use the last word separated by ';' or ','
         if key in ['A','W']:
            words = ss.split(';')
            if len(words) <= 1:
               words = ss.split(',')
            if len(words) <= 1:
               ss = ''
            else:
               ss = words[-1]
         # remove
         elif key in ['D','F','X','?']:
            ss = ''
         # keep otherwise
         #elif key in ['L','O','Q','S', 'noise']:
         m = re.sub(r'\(', r'\\(', m)
         m = re.sub(r'\)', r'\\)', m)
         m = re.sub(r'\?', r'\?', m)
         #print 'sub:', m, ss
         s = re.sub(r'%s' % m, r'%s' % ss, s)
         #print 'norm:', s

   output = ' '.join(s.split())
   # replace <SB>
   output = re.sub(r'([\?\!,\.])<SB>', r'\1', output)
   output = re.sub(r'<SB>', r'.', output)
   output = re.sub(r'<[^>]+>', r'', output)
   #print 'output:', output
   return output

def remove_tags(text):
   pattern = re.compile(r'\([^:\)]*\)')
   text = re.sub(pattern, r'', text)
   text = re.sub(ur'\)+', ur'', text, re.UNICODE)
   pattern = re.compile(ur'（[^:）]*）', re.UNICODE)
   text = re.sub(pattern, r'', text)
   text = re.sub(r'\(F[^\)]*(\)|$)', r'', text)
   text = re.sub(r'\{[^\}]*\}|<[^>]*>', r'', text)
   return text.strip()

def remove_keys(to_remove, d):
   for key in to_remove:
      if key in d:
         del d[key]
