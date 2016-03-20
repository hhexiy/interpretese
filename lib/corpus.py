# -*- coding: utf-8 -*-
import argparse, codecs, sys
from util import *
from collections import defaultdict
import numpy as np
from vw import Feat

class Word:
   def __init__(self, feats):
      n = len(feats)
      feats.extend([None]*(4-n))
      self.tok = feats[0]
      self.tag = feats[1]
      self.ctag = None
      if self.tag:
        if '_' in self.tag:
          self.ctag = self.tag.split('_')[0]
        else:
          # first two characters
          self.ctag = self.tag[:2] if len(self.tag) > 1 else self.tag
      self.stem = feats[2]
      self.reading = feats[3]
      self.is_punct = is_punct(self.tok)

class Sentence(object):
   def __init__(self, words):
      self.words = words
      self.size = len(words)
      self.nopunct_size = sum([0 if w.is_punct else 1 for w in self.words])
      self.alignments = None

   def get_chunk_alignments(self, chunks):
      """
      input: list of tuples of chunk start and end indices
      output:list of tuples of alignments for each chunk
      """
      alignments = []
      for start, end in chunks:
         align = []
         for idx in range(start, end+1):
            align.extend(self.alignments[idx])
         alignments.append(sorted(align))
      return alignments

   def get_inversion(self, chunks, mask):
      chunk_alignments = self.get_chunk_alignments(chunks)
      assert len(chunks) == len(chunk_alignments)
      invs = []
      npair = 0
      # we are looking for pairs where word j is after word i but is interpreted before i
      for i in range(len(chunks)):
         for j in range(i+1, len(chunks)):
            if not chunk_alignments[j] or not chunk_alignments[i] :
               continue
            if mask[j] == False:
               continue
            if self.has_inversion(chunk_alignments[j], chunk_alignments[i]):
               invs.append(((chunks[i], chunk_alignments[i]), (chunks[j], chunk_alignments[j])))
            npair += 1
      return invs, npair

   def has_inversion(self, alignments1, alignments2):
      """
      alignments1 should be after alignments2
      """
      if not alignments1 or not alignments2:
         return False
      if alignments1[-1] < alignments2[0]:
         return True
      return False

   def get_delay(self, mask, alignments=None):
      if not alignments:
         alignments = self.alignments
      delays = []
      prev_aligned_word = -1.0
      if not alignments:
         return delays
      for i in range(self.size):
         ai = alignments[i]
         # if filtered or not aligned or is punct, skip
         if not mask[i] or not ai or self.words[i].is_punct:
            continue
         rightmost_aligned_word = float(max(ai))
         if rightmost_aligned_word > prev_aligned_word:
            delays.append(rightmost_aligned_word - prev_aligned_word)
            prev_aligned_word = rightmost_aligned_word
      return delays

   def get_delay2(self, mask, alignments=None):
      if not alignments:
         alignments = self.alignments
      delays = []
      for i in range(self.size):
         ai = alignments[i]
         # if filtered or not aligned or is punct, skip
         if not mask[i] or not ai or self.words[i].is_punct:
            continue
         max_delay = 0
         for j in range(i+1, self.size):
            aj = alignments[j]
            if not aj:
               continue
            # all of j's aligned words are before i's
            if aj[-1] < ai[0]:
               max_delay = max(max_delay, j-i)
         delays.append(max_delay)
      # no valid word, delays is []
      return delays

   def text(self):
      return ' '.join([word.tok for word in self.words])

   def __str__(self):
      return ' '.join([u'%s|%s' % (word.tok, word.tag) for word in self.words])

   def nltk_tagged_sent(self):
      """
      list of tuples of tok and tag
      """
      return [(word.tok, word.tag) for word in self.words]

   def get_word_position(self):
      '''
      get the position of a word within its sentence
      '''
      # use period as sentence delimiter
      tags = [word.tag if word.tok != '.' else '.' for word in self.words]
      # don't include punct because get_vw_feat_ngram does not!
      #tags = ['PUNCT' if self.words[i].is_punct else tag for i, tag in enumerate(tags)]
      sents_tags = ' '.join(tags).split('.')
      pos = []
      for sent_tags in sents_tags:
         stags = sent_tags.strip().split()
         N = len(stags)
         for i, tag in enumerate(stags):
            #if tag == 'PUNCT':
            #   continue
            pos.append(bin_val(i, N))
         # for the sentence delimiter
         pos.append(1)
      #print len(tags), len(pos)
      del pos[-1]
      assert len(tags) == len(pos)
      return pos

   def get_vw_feat_ngram(self, array, namespace, upto=-1):
      # NOTE: the input array could be altered!!
      if upto > 0:
         assert upto <= 1
         upto = min(int(upto*len(array)), len(array))
         array = array[:upto]

      # interpreted text does not have quotes
      #array = [x for x in array if not is_punct(x)]

      #pos = self.get_word_position()[:upto]
      pos = None

      unigrams, unigrams_pos = get_unigrams(array, pos)
      bigrams, bigrams_pos = get_bigrams(array, pos)
      trigrams, trigrams_pos = get_trigrams(array, pos)

      unigram_str = ' '.join(unigrams)
      bigram_str = ' '.join(bigrams)
      trigram_str = ' '.join(trigrams)

      unigram_pos_str = ' '.join(unigrams_pos)
      bigram_pos_str = ' '.join(bigrams_pos)
      trigram_pos_str = ' '.join(trigrams_pos)

      feat = '|%s %s %s %s' % (namespace, unigram_str, bigram_str, trigram_str)
      feat += ' %s %s %s' % (unigram_pos_str, bigram_pos_str, trigram_pos_str)
      #feat = '|%s %s' % (namespace, unigram_str)
      return feat

   def get_vw_feat_lex(self, upto=-1):
      toks = [word.tok for word in self.words]
      #toks = [word.tok for i, word in enumerate(self.words) if i in self.alignments]
      return self.get_vw_feat_ngram(toks, 'tok', upto)

   def get_vw_feat_ctag(self, upto=-1):
      tags = [word.ctag for word in self.words]
      return self.get_vw_feat_ngram(tags, 'ctag', upto)

   def get_vw_feat_tag(self, upto=-1):
      tags = [word.tag for word in self.words]
      return self.get_vw_feat_ngram(tags, 'tag', upto)

   def get_vw_feat_lextag(self, upto=-1):
      toktags = ['%s_%s' % (word.tok, word.ctag) for word in self.words]
      #toktags = ['%s_%s' % (word.tok, word.ctag) for i, word in enumerate(self.words) if i in self.alignments]
      return self.get_vw_feat_ngram(toktags, 'toktag', upto)

   def _count_repetition(self, toks, is_content):
      d = defaultdict(int)
      for cont, tok in zip(is_content, toks):
         if cont:
            d[tok] += 1
      return d

   def get_vw_feat(self, mask, word_rank=None, upto=-1):
      '''
      mask is a tuple of booleans indication whether to use stat, tag, and lex features
      '''
      feat = []
      if mask[Feat.stat]:
         feat.append(self.get_vw_feat_stat(word_rank))
      if mask[Feat.tag]:
         feat.append(self.get_vw_feat_tag(upto))
      if mask[Feat.ctag]:
         feat.append(self.get_vw_feat_ctag(upto))
      if mask[Feat.lex]:
         feat.append(self.get_vw_feat_lex(upto))
      if mask[Feat.lextag]:
         feat.append(self.get_vw_feat_lextag(upto))
      if mask[Feat.verb]:
         feat.append(self.get_vw_feat_verb(upto))
      if mask[Feat.marker]:
         feat.append(self.get_vw_feat_marker(upto))
      return ' '.join(feat)

   def get_labeled_sent(self, chunks):
      """
      chunks is a list of tuples of chunk start and end index
      """
      labeled_sent = []
      i = 0
      while i < len(self.words):
         if i == chunks[0][0]:
            chunk = chunks[0]
            label = '**'
            chunk = None
         elif i == chunks[1][0]:
            chunk = chunks[1]
            label = '##'
         else:
            chunk = None

         if chunk:
            labeled_sent.append('[%s]-%s' % (' '.join([word.tok for word in self.words[i:chunk[-1]+1]]), label))
            i = chunk[-1] + 1
         else:
            labeled_sent.append(self.words[i].tok)
            i += 1
      return ' '.join(labeled_sent)

class JaSentence(Sentence):
   def get_num_passive(self):
      passive_num = sum([1 if word.stem.endswith(u'れる') and word.ctag == u'動詞' else 0 for word in self.words])
      return passive_num

   def get_sents(self):
      sents = ' '.join([word.tok for word in self.words]).split('.')
      new_sents = []
      for i, sent in enumerate(sents):
         s = sent.strip().split()
         if s:
            new_sents.append(s)
      return new_sents

   def is_verb(self, i):
      word = self.words[i]
      tag1, tag2 = word.tag.split('_')
      if tag1 == u'動詞' and tag2 == u'自立' and word.stem not in [u'する']:#, u'ある', u'いる']:
         return True
      if tag1 == u'名詞' and tag2 == u'サ変接続' and \
            i < len(self.words)-1 and self.words[i+1].stem == u'する':
         return True
      return False

   def is_content(self, word):
      tag1, tag2 = word.tag.split('_')
      if tag1 in [u'名詞', u'動詞', u'形容詞', u'副詞'] and tag2 not in [u'非自立', u'代名詞', u'接尾']:
         if tag1 == u'動詞' and tag2 == u'自立' and word.stem in [u'する', u'ある', u'いる']:
            return False
         return True
      else:
         return False

   def get_vw_feat_marker(self, upto=-1):
      if upto > 0:
         assert upto <= 1
         upto = min(int(upto*len(self.words)), len(self.words))
         words = self.words[:upto]
      else:
         words = self.words

      toks = [word.tok for word in words]
      tags = [word.tag if word.tok != '.' else '.' for word in words]
      sents = ' '.join(toks).split('.')
      sents_tags = ' '.join(tags).split('.')
      assert len(sents) == len(sents_tags)

      feats = []
      # topic, subject, direct object, indirect object
      marker_counts = {u'は': 0, u'が': 0, u'を': 0, u'に': 0}
      for i, tok in enumerate(toks):
         if tok in marker_counts:
            marker_counts[tok] += 1
      for tok, freq in marker_counts.items():
         feats.append('%s=%d' % (tok, freq))

      # positions in sentence
      for sent in sents:
         sent_toks = sent.strip().split()
         sent_len = len(sent_toks)
         for i, tok in enumerate(sent_toks):
            if tok in marker_counts:
               pos = bin_val(i, sent_len)
               feats.append('%s_pos=%d' % (tok, pos))

      return '|marker %s' % ' '.join(feats)

   def get_vw_feat_verb(self, upto=-1):
      if upto > 0:
         assert upto <= 1
         upto = min(int(upto*len(self.words)), len(self.words))
         words = self.words[:upto]
      else:
         words = self.words
      toks = [word.tok for word in words]
      tags = [word.tag if word.tok != '.' else '.' for word in words]
      sents = ' '.join(toks).split('.')
      sents_tags = ' '.join(tags).split('.')
      assert len(sents) == len(sents_tags)

      # verbs
      verbs = []
      for sent, tag in zip(sents, sents_tags):
         sent_toks = sent.strip().split()
         sent_tags = tag.strip().split()
         assert len(sent_toks) == len(sent_tags)

         sent_len = float(len(sent_toks))
         for i, (tok, tag) in enumerate(zip(sent_toks, sent_tags)):
            if tag == u'動詞_自立':
               verbs.append(tok)
               verbs.append('%s_pos:%f' % (tok, (i+1)/sent_len))

      return '|verb %s' % ' '.join(verbs)


   def get_vw_feat_stat(self, word_rank, no_punct):
      if no_punct:
         words = [word for word in self.words if not word.is_punct]
      else:
         words = self.words
      N = float(len(words))
      toks = [word.tok for word in words]
      tags = [tuple(word.tag.split('_')) for word in words]
      is_content = [True if tag[0] in [u'名詞', u'動詞', u'形容詞', u'副詞'] and tag[1] not in [u'非自立', u'代名詞'] else False for tag in tags]
      stems = [word.stem for word in words]
      readings = [word.reading for word in words]

      # tuples of feat name and value
      feats = []

      # mean word rank
      if word_rank:
         rank = 0
         for stem in stems:
            if stem in word_rank:
               rank += word_rank[stem]
            else:
               rank += len(word_rank)
         rank = float(rank) / N
         feats.append(('word_rank', rank))

      # lex variety
      tok_types_num = len(set(toks)) / N
      feats.append(('tok_types_num', tok_types_num))
      stem_types_num = len(set(stems)) / N
      feats.append(('stem_types_num', stem_types_num))

      # mean word length
      tok_len = sum([len(tok) for tok in toks]) / N
      feats.append(('tok_len', tok_len))

      # sent length
      sent_len = N
      feats.append(('sent_len', sent_len))

      # mean syllable length
      reading_len = sum([1 if reading == 'null' else len(reading) for reading in readings]) / N
      feats.append(('reading_len', reading_len))

      # lex density
      cont_num = sum([1 if cont else 0 for cont in is_content]) / N
      feats.append(('cont_num', cont_num))

      # proper nouns
      nnp_num = sum([1 if tag[1] == u'固有名詞' else 0 for tag in tags]) / N
      feats.append(('nnp_num', nnp_num))

      # conjunction words
      cc_num = sum([1 if tag[0] == u'接続詞' else 0 for tag in tags]) / N
      feats.append(('cc_num', cc_num))

      # pronouns
      pro_num = sum([1 if tag[1] == u'代名詞' else 0 for tag in tags]) / N
      feats.append(('pro_num', pro_num))

      # passive voice
      nverbs = sum([1 if tag[0] == u'動詞' else 0 for tag in tags])
      if nverbs == 0:
         passive_num = 0
      else:
         passive_num = sum([1 if re.search(ur'れる', word.tok, re.UNICODE) and word.tag.split('_')[0] == u'動詞' else 0 for word in words]) / float(nverbs)
      feats.append(('passive_num', passive_num))

      # repetition

      feat_str = '|stat %s' % ' '.join(['%s:%f' % (feat[0], feat[1]) for feat in feats])
      return feat_str


class EnSentence(Sentence):
   def __init__(self, words):
      self.biling_feat = None
      super(EnSentence, self).__init__(words)

   def is_verb(self, i):
      if self.words[i].tok.startswith('VB'):
         return True

   def _count_vowels(self, word):
      nvowel = 0
      prev_vowel = False
      for c in word.lower():
         if c in ['a', 'e', 'i', 'o', 'u']:
            if not prev_vowel:
               nvowel += 1
               prev_vowel = True
         else:
            prev_vowel = False
      return nvowel

   def get_sents(self):
      sent_str = self.text()

      sents = sent_tokenize(sent_str)
      sents = filter(lambda x: not is_punct(x), sents)

      # Punctuation tokenizer
      #sents = re.split(',|\.|\?|;', sent_str)
      #sents = filter(lambda x: len(x.strip().split()) > 2, sents)

      return sents

   def get_vw_feat_verb(self, upto=-1):
      return ''

   def get_vw_feat_marker(self, upto=-1):
      return ''

   def is_content(self, word):
      return True if word.ctag in ['NN', 'VB', 'JJ', 'RB'] else False

   def get_num_passive(self):
      tags = []
      for word in self.words:
         if word.tok.lower() in ['be', 'are', 'is', 'am', 'were', 'was', 'been', 'get', 'got', 'gotten', 'being']:
            tags.append('b')
         elif word.tag == 'VBN':
            tags.append('V')
         elif word.tag.startswith('VB'):
            tags.append('v')
         else:
            tags.append('0')
      tags_str = ''.join(tags)
      m = re.findall(r'b[^vV]{,4}V', tags_str)
      #if m:
      #   print self.text()
      return len(m)

   def get_word_rank(self, word_rank):
      rank = 0
      total = 0
      for word in self.words:
         if True:
            tok = word.tok
            if tok in word_rank:
               rank += float(word_rank[tok])
               total += 1
      if total == 0:
         rank = 0
      else:
         rank = rank / float(total)
      return rank

   def get_vw_feat_stat(self, word_rank, no_punct=True):
      if no_punct:
         words = [word for word in self.words if not word.is_punct]
      else:
         words = self.words
      N = float(len(words))
      toks = [word.tok.lower() for word in words]
      tags = [word.tag for word in words]
      ctags = [word.ctag for word in words]
      # noun, verb, adjective, adverb
      is_content = [1 if self.is_content(word) else 0 for word in words]
      stems = [stemmer.stem(tok) for tok in toks]

      # tuples of feat name and value
      feats = []

      # mean word rank
      if word_rank:
         rank = 0
         total = 0
         for tok in toks:
            if tok in word_rank:
               rank += word_rank[tok]
               total += 1
         if total == 0:
            rank = 0
         else:
            rank = rank / float(total)
         feats.append(('unigram_freq', rank))

         rank = 0
         total = 0
         for i in range(len(toks)-1):
            tok = '%s %s' % (toks[i], toks[i+1])
            if tok in word_rank:
               rank += word_rank[tok]
               total += 1
         if total == 0:
            rank = 0
         else:
            rank = rank / float(total)
         feats.append(('bigram_freq', rank))

      # lex variety
      tok_types_num = len(set(toks)) / N
      feats.append(('tok_types_num', tok_types_num))
      stem_types_num = len(set(stems)) / N
      feats.append(('stem_types_num', stem_types_num))

      # pronouns
      npro = 0
      for tok, tag in zip(toks, tags):
         if tok.lower() in ['this', 'that', 'these', 'those'] and tag == 'DT':
            npro += 1
      feats.append(('prodt_num', npro / N))

      # mean word length
      tok_len = sum([len(tok) for tok in toks]) / N
      feats.append(('tok_len', tok_len))

      # mean syllable length (approximated by number of vowels)
      num_vowels = sum([self._count_vowels(tok) for tok in toks]) / N
      feats.append(('sylb_len', num_vowels))

      # sent length
      sent_len = N
      feats.append(('sent_len', sent_len))

      # lex density
      # TODO: normalize by source length
      cont_num = sum(is_content)
      feats.append(('cont_num', cont_num))
      feats.append(('cont_ratio', cont_num/N))

      # proper nouns
      nnp_num = sum([1 if tag == 'NNP' else 0 for tag in tags]) / N
      feats.append(('nnp_num', nnp_num))

      # conjunction words
      cc_num = sum([1 if tag == 'CC' else 0 for tag in tags]) / N
      feats.append(('cc_num', cc_num))

      # pronouns
      pro_num = sum([1 if tag[:2] == 'PR' else 0 for tag in tags]) / N
      feats.append(('pro_num', pro_num))

      # passive voice
      passive_num = self.get_num_passive()
      nverbs = sum([1 if ctag == 'VB' else 0 for ctag in ctags])
      if nverbs == 0:
         feats.append(('passive_num', 0))
      else:
         feats.append(('passive_num', passive_num / float(nverbs)))

      # repetition
      repeat_dict = self._count_repetition(toks, is_content)
      repeated_word_freqs = [freq for freq in repeat_dict.values() if freq > 1]
      repeat_word_num = len(repeated_word_freqs) / N
      feats.append(('repeat_word_num', repeat_word_num))
      if len(repeated_word_freqs) == 0:
         repeat_times = 0
      else:
         repeat_times = sum(repeated_word_freqs) / float(len(repeated_word_freqs))
      feats.append(('repeat_times', repeat_times))

      # segment number
      nsegs = max(1, len(self.get_sents())) / float(self.nopunct_size)
      feats.append(('num_segs', nsegs))

      # inversions
      for feat_name, feat_val in self.biling_feat.items():
         feats.append((feat_name, feat_val))

      feat_stat_str = '|stat %s' % ' '.join(['%s:%f' % (feat[0], feat[1]) for feat in feats])

      return feat_stat_str


class SentencePair:
   def __init__(self, src_sent, tgt_sent, alignments=None):
      self.src_sent = src_sent
      self.tgt_sent = tgt_sent
      # alignments: [src_idx] = [tgt indices]
      self.alignments = alignments

      # check alignment
      # length does not match or too few alignments are not good
      # TODO: this is probably not a good heuristic as there are summarization etc.
      #self.good_alignment = True
      #len_ratio = float(src_sent.size) / float(tgt_sent.size)
      #if len_ratio < 0.5 or len_ratio > 2:
      #   self.good_alignment = False
      #   if alignments:
      #      num_nopunct_alignment = sum([1 if not is_punct(src_sent.words[i].tok) else 0 for i in alignments])
      #      if src_sent.nopunct_size > num_nopunct_alignment * 3:
      #         self.good_alignment = False
      #   else:
      #      self.good_alignment = False

      if self.alignments:
         self.good_alignment = True
      else:
         self.good_alignment = False

      # get tgt_alignments
      self.tgt_alignments = None
      if alignments:
         tgt_alignments = defaultdict(list)
         for src_idx, tgt_ind in self.alignments.items():
            for tgt_idx in tgt_ind:
               tgt_alignments[tgt_idx].append(src_idx)
         for tgt_idx in tgt_alignments:
            tgt_alignments[tgt_idx].sort()
         self.tgt_alignments = tgt_alignments

      self.src_sent.alignments = self.alignments
      self.tgt_sent.alignments = self.tgt_alignments

   def get_tgt_in_src_order(self):
      words = []
      for src_idx in range(self.src_sent.size):
         if src_idx in self.src_sent.alignments:
            words.extend([self.tgt_sent.words[i].tok for i in self.src_sent.alignments[src_idx]])
      return words

   def compute_bilingual_feats(self):
      '''
      TODO: have to have this now for alignment features...
      '''
      feat = {}
      sent = self.src_sent
      chunks = [(i, i) for i in range(len(sent.words)) if not sent.words[i].is_punct]

      mask = [True] * len(chunks)
      invs, npairs = sent.get_inversion(chunks, mask)
      feat['num_invs'] = 0 if npairs == 0 else len(invs) / float(npairs)

      num_align = sum([1 if not self.src_sent.words[i].is_punct else 0 for i in self.alignments]) / float(self.src_sent.nopunct_size)
      feat['num_align'] = num_align

      self.tgt_sent.biling_feat = feat

   def get_passive_aligned_sent(self):
      '''
      get src sentence that get passivized in tgt
      '''
      m = self.tgt_sent.get_num_passive()
      if not m:
         return None
      # m returned interval is [,,)
      tgt_verb_ind = [x.end(0)-1 for x in m]
      src_sents = self.src_sent.get_sents()
      sents_len = list(np.cumsum([len(sent) for sent in src_sents]))
      src_sent_ind = []
      for verb_idx in tgt_verb_ind:
         if self.tgt_alignments and verb_idx in self.tgt_alignments:
            src_word_ind = self.tgt_alignments[verb_idx]
            for src_word_idx in src_word_ind:
               for i, l in enumerate(sents_len):
                  if src_word_idx < l:
                     src_sent_ind.append(i)
                     break
      if not src_sent_ind:
         return None
      return sorted(list(set(src_sent_ind)))

   def get_possesive_alignments(self):
      # only support Japanese to English now
      assert self.src_sent.__class__.__name__ == 'JaSentence'
      num_of = 0   # aligned to of_IN
      num_s = 0    # aligned to 's_POS
      num_src_pos = 0
      for src_idx, tgt_ind in self.alignments.items():
         if self.src_sent.words[src_idx].tok == u'の':
            num_src_pos += 1
            for tgt_idx in tgt_ind:
               tgt_tok = self.tgt_sent.words[tgt_idx].tok
               if tgt_tok == "'s":
                  num_s += 1
                  break
               elif tgt_tok == 'of':
                  num_of += 1
                  break
      return num_src_pos, num_of, num_s


   def get_omission(self, lang):
      '''
      get omitted content word/tag in the source/target text
      '''
      omitted_words = []
      if lang == 'src':
         for i, word in enumerate(self.src_sent.words):
            if self.src_sent.is_content(word) and i not in self.alignments:
               omitted_words.append(word)
      else:
         for i, word in enumerate(self.tgt_sent.words):
            if i not in self.alignments:
               omitted_words.append(word)
      return omitted_words

   def __str__(self):
      try:
         s = ['%s_%s|%d-%s|%s' % (self.src_sent.words[src_word_idx].tok, self.src_sent.words[src_word_idx].tag, src_word_idx, ' '.join(['%s_%s' % (self.tgt_sent.words[tgt_word_idx].tok, self.tgt_sent.words[tgt_word_idx].tag) for tgt_word_idx in tgt_trans_idx]), ','.join([str(x) for x in tgt_trans_idx])) for (src_word_idx, tgt_trans_idx) in sorted(self.alignments.items())]
         return '\t'.join(s).encode('utf-8')
      except UnicodeDecodeError:
         return '\n'

class ParallelCorpus:
   def __init__(self, src_file, src_lang, tgt_file, tgt_lang, align_file=None):
      self.src_lang = src_lang
      self.tgt_lang = tgt_lang
      src_sents = self.read_sents(src_file, src_lang)
      tgt_sents = self.read_sents(tgt_file, tgt_lang)
      if align_file:
         # alignment is a dict
         alignments = self.read_alignments(align_file)
         # remove punct alignments
         for i, (src_sent, alignment) in enumerate(zip(src_sents, alignments)):
            punct_alignments = []
            for src_idx in alignment:
               if is_punct(src_sent.words[src_idx].tok):
                  punct_alignments.append(src_idx)
            if punct_alignments:
               remove_keys(punct_alignments, alignments[i])
         # make pair
         self.sent_pairs = [SentencePair(src_sent, tgt_sent, alignment) for src_sent, tgt_sent, alignment in zip(src_sents, tgt_sents, alignments)]
      else:
         self.sent_pairs = [SentencePair(src_sent, tgt_sent) for src_sent, tgt_sent in zip(src_sents, tgt_sents)]

   def get_tgt_sent_len(self):
      return sum([sent_pair.ja_sent.nopunct_size for sent_pair in self.sent_pairs])

   def print_tgt_in_src_order(self, filename):
      with open(filename, 'w') as fout:
         for sent_pair in self.sent_pairs:
            words = []
            for src_idx in range(sent_pair.src_sent.size):
               if src_idx in sent_pair.src_sent.alignments:
                  words.extend([sent_pair.tgt_sent.words[i].tok for i in sent_pair.src_sent.alignments[src_idx]])
            fout.write(sent_pair.tgt_sent.text() + '\n')
            fout.write(' '.join(words) + '\n\n')

   def get_bad_alignments(self, fname=None):
      bad_alignments = []
      for sent_pair in self.sent_pairs:
         if not sent_pair.good_alignment:
            bad_alignments.append(' '.join([word.tok for word in sent_pair.en_sent.words]))
      if fname:
         with codecs.open(fname, 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(bad_alignments))
      return bad_alignments

   def get_tgt_vocab(self, fname=None):
      vocab = defaultdict(int)
      for sent_pair in self.sent_pairs:
         for word in sent_pair.tgt_sent.words:
            # Japanese vocabulary
            if self.tgt_lang == 'ja':
               coarse_tag = word.tag.split('_')[0]
               if coarse_tag in [u'名詞', u'動詞', u'形容詞', u'副詞'] and (not is_punct(word.tok)) and (not re.match(ur'[0-9]+', word.tok, re.UNICODE)):
                  vocab[word.stem] += 1
            # English vocabulary
            else:
               coarse_tag = word.tag[0]
               # TODO: check this
               if coarse_tag in ['N', 'V', 'A']:
                  vocab[word.tok] += 1
      if fname:
         with codecs.open(fname, 'w', encoding='utf-8') as fout:
            for tok, freq in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
               fout.write('%s\t%d\n' % (tok, freq))
      return vocab

   def read_sents(self, fname, lang):
      sents = []
      with codecs.open(fname, 'r', encoding='utf-8') as fin:
         for line in fin:
            words = line.strip().split()
            if lang == 'en':
               sents.append(EnSentence([Word(feats) for feats in map(lambda x: x.split('|'), words)]))
            elif lang == 'ja':
               sents.append(JaSentence([Word(feats) for feats in map(lambda x: x.split('|'), words)]))
            else:
               raise ValueError("Unknown language.")

      return sents

   def get_src_word_alignment(self, line):
      """
      alignment file format:
      foreign_word_idx - en_word_idx
      """
      alignments = defaultdict(list)
      for pair in line.split():
         fr_idx, en_idx = [int(x) for x in pair.split('-')]
         # source is english
         if self.src_lang == 'en':
            alignments[en_idx].append(fr_idx)
         # source is foreigh
         else:
            alignments[fr_idx].append(en_idx)
      for src_idx in alignments:
         alignments[src_idx].sort()
      return alignments

   def read_alignments(self, align_file):
      alignments = []
      with open(align_file, 'r') as fin:
         for line in fin:
            alignment = self.get_src_word_alignment(line.strip())
            alignments.append(alignment)
      return alignments

   def print_alignments(self, fname=None):
      if not fname:
         fout = codecs.getwriter('utf-8')(sys.stdout)
      else:
         #fout = codecs.open(fname, 'w', encoding='utf-8', errors='ignored')
         fout = open(fname, 'w')
      for sent_pair in self.sent_pairs:
         fout.write('%s\n' % sent_pair)
      if fname:
         fout.close()

   def get_num_align(self):
      return sum([len(sent_pair.alignments) for sent_pair in self.sent_pairs])

