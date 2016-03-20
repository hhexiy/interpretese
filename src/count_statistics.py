# -*- coding: utf-8 -*-
from util import *
import math
import argparse, codecs, sys, re
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy import stats
from vocabulary import *
from corpus import ParallelCorpus
from vw import *
import bleu
import cPickle

scratch='/tmp'

def filter_by_alignments(mt_para_corpus, si_para_corpus):
   N = len(mt_para_corpus.sent_pairs)
   nfiltered = 0
   for i in range(N-1, -1, -1):
      if not mt_para_corpus.sent_pairs[i].good_alignment or \
         not si_para_corpus.sent_pairs[i].good_alignment:
         del mt_para_corpus.sent_pairs[i]
         del si_para_corpus.sent_pairs[i]
         nfiltered += 1
   print 'filtered:', nfiltered

def ttest(list1, list2):
   a1 = np.array(list1)
   a2 = np.array(list2)
   t, prob = stats.ttest_rel(a1, a2)
   print '-'*40
   print '{:<10s}{:<10s}{:<10s}{:<10s}'.format('mean1', 'mean2', 't-stat', 'p-value')
   print '{:<10.6f}{:<10.6f}{:<10.6f}{:<10.6f}'.format(np.mean(a1), np.mean(a2), t, prob)
   print '-'*40

def intersect_alignments(mt_para_corpus, si_para_corpus):
   for mt_sent_pair, si_sent_pair in zip(mt_para_corpus.sent_pairs, si_para_corpus.sent_pairs):
      mt_keys = set(mt_sent_pair.alignments.keys())
      si_keys = set(si_sent_pair.alignments.keys())
      common_keys = mt_keys & si_keys
      mt_to_remove = mt_keys - common_keys
      si_to_remove = si_keys - common_keys
      remove_keys(mt_to_remove, mt_sent_pair.alignments)
      remove_keys(si_to_remove, si_sent_pair.alignments)

def count_inversion(para_corpus, tag=None):
   num_inv = 0
   num_pair = 0
   total_inv_dist = 0
   # for t-test:
   # number of invs in each sentence *after intersect alignments*, i.e. each sentence has the same number of aligned word in MT and SI
   sent_invs = []
   sent_inv_dist = []
   all_invs = []
   for sent_pair in para_corpus.sent_pairs:
      chunks = [(i, i) for i in range(len(sent_pair.src_sent.words)) if not sent_pair.src_sent.words[i].is_punct]
      # TODO: chunk with multiple words
      if tag:
         mask = [True if sent_pair.src_sent.words[chunk[0]].ctag == tag else False for chunk in chunks]
      else:
         mask = [True] * len(chunks)
      if len(chunks) == 0:
         #print sent_pair.src_sent.__str__().encode('utf-8')
         continue

      invs, npairs = sent_pair.src_sent.get_inversion(chunks, mask)
      num_pair += npairs
      num_inv += len(invs)
      sent_invs.append(len(invs))
      all_invs.append(invs)

      # dist is the inversion distance in source
      inv_dist = 0
      for inv in invs:
         d = inv[1][0][0] - inv[0][0][0]
         assert d > 0
         inv_dist += d
      total_inv_dist += inv_dist
      sent_inv_dist.append(inv_dist)

   print 'percentage of inversion: %d/%d=%f' % (num_inv, num_pair, float(num_inv) / num_pair)
   print 'average inversion distance: %d/%d=%f' % (total_inv_dist, num_pair, float(total_inv_dist) / num_pair)
   #print sent_invs
   return sent_invs, sent_inv_dist, all_invs

def plot_inversion(inv_dist, sent_len, title, fname):
   fig = plt.figure()
   ax = fig.add_subplot(111)
   ax.set_xlabel('Sentence Length')
   ax.set_ylabel('Inversion Distance')
   ax.set_title(title)
   ax.plot(inv_dist, sent_len, 'bo')
   fig.savefig(fname, format='pdf')

def get_tag_masks(en_tags, en_align, pattern=None):
   masks = []
   for align, tags in zip(en_align, en_tags):
      masks.append([True if (pattern is None or re.match(pattern, tags[a[0]])) else False for a in align])
   return masks

def bleu_score(mt_para_corpus, si_para_corpus, N=4):
   '''
   BLEU score between trans and inter
   '''
   stats = [0 for i in xrange(10)]
   for mt_sent_pair, si_sent_pair in zip(mt_para_corpus.sent_pairs, si_para_corpus.sent_pairs):
      ref = [w.tok for w in mt_sent_pair.tgt_sent.words]
      output = [w.tok for w in si_sent_pair.tgt_sent.words]
      stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(output,ref))]
   return bleu.bleu(stats)

def human_consistency(src_sents):
   '''
   src_sents is a dictionary:
   src sentence -> human labels (>=1)
   ninst: total number of instances (repeated and non-repeated)
   '''
   pos_consistency = []
   neg_consistency = []
   nrepeated = []
   ncorrect = 0
   total = 0
   tp = 0
   pp = 0
   rp = 0
   for sent, values in src_sents.items():
      # calculate majority vote accuracy (upper bound)
      pos = sum(values)
      neg = len(values) - pos
      rp += pos
      if pos > neg:
         # predict positive
         ncorrect += pos
         pp += len(values)
         tp += pos
      else:
         ncorrect += neg
      total += len(values)

      if len(values) > 1:
         nrepeated.append(len(values))
         if 1 in values:
            pos_consistency.append(pos / float(len(values)))
         if 0 in values:
            neg_consistency.append(neg / float(len(values)))

   mv_acc = ncorrect / float(total)
   precision = float(tp) / pp
   recall = float(tp) / rp
   fscore = 2 * (precision*recall) / (precision+recall)

   print 'number of repeated sents:', len(nrepeated), mean(nrepeated)
   print 'human consistency (pos):', mean(pos_consistency)
   print 'human consistency (neg):', mean(neg_consistency)
   print 'majority vote accuracy:', mv_acc
   print 'majority vote fscore:', fscore

def compare_passive(mt_para_corpus, si_para_corpus):
   src_npass = [sent_pair.src_sent.get_num_passive() for sent_pair in mt_para_corpus.sent_pairs]
   mt_npass = [sent_pair.tgt_sent.get_num_passive() for sent_pair in mt_para_corpus.sent_pairs]
   mt_nverbs = [sum([1 if word.ctag == 'VB' else 0 for word in sent_pair.tgt_sent.words]) for sent_pair in mt_para_corpus.sent_pairs]
   si_npass = [sent_pair.tgt_sent.get_num_passive() for sent_pair in si_para_corpus.sent_pairs]
   si_nverbs = [sum([1 if word.ctag == 'VB' else 0 for word in sent_pair.tgt_sent.words]) for sent_pair in si_para_corpus.sent_pairs]
   #print src_npass
   #print mt_npass
   #print si_npass

   # conditional matrix
   src_pass = [[], []]
   num_src_pass = 0
   src_nopass = [[], []]
   num_src_nopass = 0
   labels = []
   # write passivized sentences
   #fout = open('je.passive.txt', 'w')
   # check interpreter's behavior for the same sentence
   src_sents = defaultdict(list)
   for i, (src, mt, si, mt_nverb, si_nverb) in enumerate(zip(src_npass, mt_npass, si_npass, mt_nverbs, si_nverbs)):
      mt /= (1 if float(mt_nverb) == 0 else float(mt_nverb))
      si /= (1 if float(si_nverb) == 0 else float(si_nverb))
      if src > 0:
         src_pass[0].append(mt)
         src_pass[1].append(si)
         num_src_pass += 1
         labels.append(0)
      else:
         src_nopass[0].append(mt)
         src_nopass[1].append(si)
         num_src_nopass += 1
         labels.append(1 if si > 0 else -1)
         src_sent = mt_para_corpus.sent_pairs[i].src_sent.text()
         src_sents[src_sent].append(1 if labels[-1] == 1 else 0)
         #if labels[-1] == 1:
         #   fout.write('%s\n' % si_para_corpus.sent_pairs[i].tgt_sent.text())
   #fout.close()

   # human consistency
   human_consistency(src_sents)

   print 'src not passive (%d) (si vs mt):' % num_src_nopass
   ttest(src_nopass[1], src_nopass[0])
   print 'src passive (%d) (si vs mt):' % num_src_pass
   ttest(src_pass[1], src_pass[0])
   return

def count_passive(para_corpus, comp_para_corpus, fname=None):
   num_passive = 0
   num_sent = 0
   if fname:
      fout = codecs.open(fname, 'w', encoding='utf-8')
   for sent_pair, comp_sent_pair in zip(para_corpus.sent_pairs, comp_para_corpus.sent_pairs):
      for i, word in enumerate(sent_pair.ja_sent.words):
         if re.search(ur'れる', word.tok, re.UNICODE) and word.tag == u'動詞':
            num_passive += 1
            if fname:
               fout.write('%s\n' % sent_pair.en_sent)
               fout.write('%s\n' % sent_pair.ja_sent.get_labeled_sent([(i, i), (i, i)]))
               fout.write('%s\n' % comp_sent_pair.ja_sent)
      num_sent += 1
   if fname:
      fout.close()
   print 'passive voice percentage: %d/%d = %f' % (num_passive, num_sent, float(num_passive)/num_sent)

def count_mtu(para_corpora):
   total_len = 0
   num = 0
   for sent_pair in para_corpora.sent_pairs:
      if sent_pair.good_alignment:
         mtus = sent_pair.get_mtus()
         # really long sentence can get empty alignment
         if not mtus:
            continue
         # En segment length
         total_len += sum([u[0][1] - u[0][0] + 1 for u in mtus])
         num += len(mtus)
   print 'average MTU length: %d/%d=%f' % (total_len, num, total_len / float(num))
   return

def get_omission_weights(mt_para_corpus, si_para_corpus, lang):
   '''
   get tf-idf weights
   '''
   tag_weights = defaultdict(int)
   tok_weights = defaultdict(int)
   N = 0  # number of docs
   for mt_sent_pair, si_sent_pair in zip(mt_para_corpus.sent_pairs, si_para_corpus.sent_pairs):
      if mt_sent_pair.good_alignment and si_sent_pair.good_alignment:
         N += 2
         for sent_pair in [mt_sent_pair, si_sent_pair]:
            tagset = set()
            tokset = set()
            for word in sent_pair.get_omission(lang):
               tagset.add(word.tag)
               tokset.add(word.tok)
            for tag in tagset:
               tag_weights[tag] += 1
            for tok in tokset:
               tok_weights[tok] += 1

   for tag, n in tag_weights.items():
      tag_weights[tag] = math.log(N/n)
   #print '\n'.join(['%s %f' % (tag, weight) for tag, weight in tag_weights.items()])
   for tok, n in tok_weights.items():
      tok_weights[tok] = math.log(N/n)
   return tag_weights, tok_weights

def count_omission(mask, para_corpus, tag_weights, tok_weights, lang):
   '''
   mask: for skipping examples
   '''
   # omission counts for each tag category
   omit_tag = defaultdict(float)
   omit_tok = defaultdict(float)
   # omission counts for each tag category and for each sentence
   omit_detail = defaultdict(list)
   # overall omission counts
   omit_all = []
   for m, sent_pair in zip(mask, para_corpus.sent_pairs):
      if m:
         # init for this sentence
         for tag in tag_weights:
            omit_detail[tag].append(0)
         N = sent_pair.src_sent.nopunct_size if lang == 'src' else sent_pair.tgt_sent.nopunct_size
         num_omit = 0
         for word in sent_pair.get_omission(lang):
            num_omit += 1
            omit_tag[word.tag] += tag_weights[word.tag] / N
            omit_tok[word.tok] += tok_weights[word.tok] / N
            # inc for this sentence
            omit_detail[word.tag][-1] += tag_weights[word.tag] / N
         omit_all.append(float(num_omit) / N)

   # sort omission based on tf-idf scores
   omit_tag = sorted(omit_tag.items(), key=lambda x: x[1], reverse=True)
   omit_tok = sorted(omit_tok.items(), key=lambda x: x[1], reverse=True)
   return omit_tag, omit_detail, omit_tok, omit_all

def compare_word_rank(mt_para_corpus, si_para_corpus, word_rank):
   mt_word_rank = []
   si_word_rank = []
   for mt_sent_pair, si_sent_pair in zip(mt_para_corpus.sent_pairs, si_para_corpus.sent_pairs):
       mt_word_rank.append(mt_sent_pair.tgt_sent.get_word_rank(word_rank))
       si_word_rank.append(si_sent_pair.tgt_sent.get_word_rank(word_rank))
   print 'word frequency (si vs mt):'
   ttest(si_word_rank, mt_word_rank)

def compare_omission(mt_para_corpus, si_para_corpus, lang):
   tag_weights, tok_weights = get_omission_weights(mt_para_corpus, si_para_corpus, lang)

   mask = []
   for mt_sent_pair, si_sent_pair in zip(mt_para_corpus.sent_pairs, si_para_corpus.sent_pairs):
      if mt_sent_pair.good_alignment and si_sent_pair.good_alignment:
         mask.append(True)
      else:
         mask.append(False)
   mt_omit, mt_omit_detail, mt_omit_tok, mt_omit_all = count_omission(mask, mt_para_corpus, tag_weights, tok_weights, lang)
   si_omit, si_omit_detail, si_omit_tok, si_omit_all = count_omission(mask, si_para_corpus, tag_weights, tok_weights, lang)

   top_k = 10
   print 'overall omission (si vs mt):'
   ttest(si_omit_all, mt_omit_all)

   print 'MT tag omissions:'
   print u'\n'.join(['%s\t%f' % (x[0], x[1]) for x in mt_omit if tag_weights[x[0]] > 0]).encode('utf-8')
   print u'MT tok omissions:'
   print u'\n'.join(['%s\t%f' % (x[0], x[1]) for x in mt_omit_tok[:top_k] if tok_weights[x[0]] > 0]).encode('utf8')
   print 'SI tag omissions:'
   print u'\n'.join(['%s\t%f' % (x[0], x[1]) for x in si_omit if tag_weights[x[0]] > 0]).encode('utf8')
   print 'SI tok omissions:'
   print u'\n'.join(['%s\t%f' % (x[0], x[1]) for x in si_omit_tok[:top_k] if tok_weights[x[0]] > 0]).encode('utf8')

   print 'Sentence omission stats:'
   for tag in tag_weights:
      if tag_weights[tag] > 0:
         mt_mean = sum(mt_omit_detail[tag])
         si_mean = sum(si_omit_detail[tag])
         t, prob = stats.ttest_rel(mt_omit_detail[tag], si_omit_detail[tag])
         if prob < 0.05:
            print (u'%s\t%f\t%f\t%f\t%f' % (tag, mt_mean, si_mean, t, prob)).encode('utf8')

def count_num_sents(sents):
   '''
   sent: an instance of Sentence
   '''
   nsents = []
   # uni, bi, tri grams
   N = 5
   sent_prefix = [defaultdict(lambda: defaultdict(int)) for i in range(N)]
   for sent in sents:
      seg_sents = sent.get_sents()
      nsents.append(max(1, len(seg_sents)) / float(sent.size))
      #nsents.append(max(1, len(seg_sents)))

      for j, seg_sent in enumerate(seg_sents):
         toks = seg_sent.split()
         for i in range(N):
            prefix = ' '.join(toks[:min(i+1, len(toks))])
            if j == 0:
               sent_prefix[i][0][prefix] += 1
            else:
               sent_prefix[i][1][prefix] += 1

   return nsents, sent_prefix

def compare_segments(mt_para_corpus, si_para_corpus):
   mt_text = [sent_pair.tgt_sent for sent_pair in mt_para_corpus.sent_pairs]
   si_text = [sent_pair.tgt_sent for sent_pair in si_para_corpus.sent_pairs]
   mt_nsents, mt_sent_prefix = count_num_sents(mt_text)
   si_nsents, si_sent_prefix = count_num_sents(si_text)

   # sent prefix
   # check prefix of segmented sentences
   for i in range(3):
      mt_sent_prefix_sorted_start = sorted(mt_sent_prefix[i][0].items(), key=lambda x: x[1], reverse=True)
      mt_sent_prefix_sorted_mid = sorted(mt_sent_prefix[i][1].items(), key=lambda x: x[1], reverse=True)
      si_sent_prefix_sorted_start = sorted(si_sent_prefix[i][0].items(), key=lambda x: x[1], reverse=True)
      si_sent_prefix_sorted_mid = sorted(si_sent_prefix[i][1].items(), key=lambda x: x[1], reverse=True)
      for fname, sent_prefix in zip( \
            ['mt_prefix_%d_start.txt' % i, 'mt_prefix_%d_mid.txt' % i, \
            'si_prefix_%d_start.txt' % i, 'si_prefix_%d_mid.txt' % i], \
            [mt_sent_prefix_sorted_start, mt_sent_prefix_sorted_mid, \
            si_sent_prefix_sorted_start, si_sent_prefix_sorted_mid]):
         with codecs.open(fname, 'w', encoding='utf8') as fout:
            for prefix, freq in sent_prefix:
               fout.write('%s %d\n' % (prefix, freq))

   # check interpreter's behavior for the same sentence
   src_sents = defaultdict(list)
   for i, (mt_nsent, si_nsent) in enumerate(zip(mt_nsents, si_nsents)):
      src_sent = mt_para_corpus.sent_pairs[i].src_sent.text()
      src_sents[src_sent].append(1 if si_nsent > mt_nsent else 0)

   # human consistency
   human_consistency(src_sents)

   # stat
   print 'Average number of sentences per chunk (si vs mt):'
   ttest(si_nsents, mt_nsents)
   print 'number of SI chunks having more sentences than MT:', sum([1 if si > mt else 0 for mt, si in zip(mt_nsents, si_nsents)])

   return

   # plot
   src_ntoks = [len(sent_pair.src_sent.words) for sent_pair in mt_para_corpus.sent_pairs]
   assert len(src_ntoks) == len(mt_nsents)
   assert len(src_ntoks) == len(si_nsents)
   mt_nsents_dict = defaultdict(list)
   si_nsents_dict = defaultdict(list)
   for src_ntok, mt_nsent, si_nsent in zip(src_ntoks, mt_nsents, si_nsents):
      mt_nsents_dict[src_ntok].append(mt_nsent)
      si_nsents_dict[src_ntok].append(si_nsent)

   x = sorted(set(src_ntoks))
   #for i in x:
   #   print i, mt_nsents_dict[i], si_nsents_dict[i]
   y_mt = [sum(mt_nsents_dict[i]) / float(len(mt_nsents_dict[i])) for i in x]
   y_si = [sum(si_nsents_dict[i]) / float(len(si_nsents_dict[i])) for i in x]
   # ratio
   y = [y_si[i]/y_mt[i] for i in range(len(y_mt))]

   fig = plt.figure()
   ax = fig.add_subplot(111)
   ax.set_xlabel('# of src tokens')
   ax.set_ylabel('# of tgt sentences')
   #ax.plot(x, y_mt, marker='o', color='b', label='MT')
   #ax.plot(x, y_si, marker='o', color='r', label='SI')
   #ax.plot(x, y_mt, 'bo')
   #ax.plot(x, y_si, 'ro')
   ax.plot(x, y, 'ro')
   #ax.legend(loc=2)
   fig.savefig('segment.pdf', format='pdf')

def compare_inversions(mt_para_corpus, si_para_corpus, tag=None, result=None):
   if not tag:
      print 'All inversions:'
   else:
      print ('%s inversions:' % tag).encode('utf-8')
   print 'MT total inversion (%):'
   mt_invs, mt_inv_dist, mt_inv_chunks = count_inversion(mt_para_corpus, tag)
   print 'SI total inversion (%):'
   si_invs, si_inv_dist, si_inv_chunks = count_inversion(si_para_corpus, tag)
   ttest(si_invs, mt_invs)

   if result:
      with codecs.open(result, 'w', encoding='utf-8') as fout:
         mt_sents = [sent_pair.tgt_sent for sent_pair in mt_para_corpus.sent_pairs]
         si_sents = [sent_pair.tgt_sent for sent_pair in si_para_corpus.sent_pairs]
         for mt_inv, si_inv, mt_sent, si_sent, mt_inv_chunk, si_inv_chunk in zip(mt_invs, si_invs, mt_sents, si_sents, mt_inv_chunks, si_inv_chunks):
            if mt_inv < si_inv:
               #fout.write('SI:\t%s\n' % ' '.join([word.tok for word in si_sent.words]))
               if len(si_inv_chunk) > 0:
                  chunk = [si_inv_chunk[0][0][1], si_inv_chunk[0][1][1]]
                  fout.write('SI:\t%s\n' % si_sent.get_labeled_sent(chunk))
               else:
                  fout.write('SI:\t%s\n' % ' '.join([word.tok for word in si_sent.words]))
               #for inv_chunk in si_inv_chunk:
               #   chunk = [inv_chunk[0][1], inv_chunk[1][1]]
               #   fout.write('  :\t%s\n' % si_sent.get_labeled_sent(chunk))

               #fout.write('MT:\t%s\n' % ' '.join([word.tok for word in mt_sent.words]))
               if len(mt_inv_chunk) > 0:
                  chunk = [mt_inv_chunk[0][0][1], mt_inv_chunk[0][1][1]]
                  fout.write('MT:\t%s\n' % mt_sent.get_labeled_sent(chunk))
               else:
                  fout.write('MT:\t%s\n' % ' '.join([word.tok for word in mt_sent.words]))
               #for inv_chunk in mt_inv_chunk:
               #   chunk = [inv_chunk[0][1], inv_chunk[1][1]]
               #   fout.write('  :\t%s\n' % mt_sent.get_labeled_sent(chunk))

               fout.write('\n')

def compare_length(mt_para_corpora, si_para_corpora, th=0.3):
   with codecs.open('si.long.txt', 'w', 'utf8') as fout_long, codecs.open('si.short.txt', 'w', 'utf8') as fout_short:
      nshort = 0
      nlong = 0
      for mt_sent_pair, si_sent_pair in zip(mt_para_corpora.sent_pairs, si_para_corpora.sent_pairs):
         if mt_sent_pair.good_alignment and si_sent_pair.good_alignment:
            si_sent = si_sent_pair.tgt_sent
            mt_sent = mt_sent_pair.tgt_sent
            len_ratio = si_sent.nopunct_size / float(mt_sent.nopunct_size)
            if len_ratio < (1-th):
               fout_short.write('%s\n%s\n\n' % (si_sent.text().encode('utf8'), mt_sent.text()))
               nshort += 1
            elif len_ratio > (1+th):
               fout_long.write('%s\n%s\n\n' % (si_sent.text().encode('utf8'), mt_sent.text()))
               nlong += 1
      print 'number of SI sentence shorter than MT:', nshort
      print 'number of SI sentence longer than MT:', nlong


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--src', dest='src', action='store', type=str, help='source language')
   parser.add_argument('--tgt', dest='tgt', action='store', type=str, help='target language')
   parser.add_argument('--src_file', dest='src_file', action='store', type=str, help='source sentences (tagged)')
   parser.add_argument('--mt_file', dest='mt_file', action='store', type=str, help='MT sentences (tagged)')
   parser.add_argument('--si_file', dest='si_file', action='store', type=str, help='SI sentences (tagged)')
   parser.add_argument('--mt_align_file', dest='mt_align_file', help='output of berkeley aligner for MT')
   parser.add_argument('--si_align_file', dest='si_align_file', help='output of berkeley aligner for SI')
   parser.add_argument('--invert', dest='invert', action='store_true', help='count NP inversion')
   parser.add_argument('--vocab', dest='vocab', action='store_true', help='count vocabulary')
   parser.add_argument('--passive', dest='passive', action='store_true',  help='count passive voice')
   parser.add_argument('--omission', dest='omission', action='store_true',  help='count omitted english words/sentences')
   parser.add_argument('--vw', dest='vw', action='store_true',  help='train vw ngram model')
   parser.add_argument('--bleu', dest='bleu_N', action='store', type=int, help='get bleu score (N=)')
   parser.add_argument('--ngram_freq', dest='ngram_freq', action='store', help='ngram frequency')
   parser.add_argument('--align', dest='align', type=str, help='file to print aligned words')
   parser.add_argument('--segment', dest='segment', action='store_true', help='analyze sentence segment')
   parser.add_argument('--word_rank', action='store_true', help='analyze word rank')
   parser.add_argument('--length', dest='length', action='store_true', help='analyze sentence length')
   parser.add_argument('--stat', dest='stat', action='store_true', help='get corpus statistics')
   args = parser.parse_args()

   # external corpus ngram frequency
   word_rank = None
   if args.ngram_freq:
      # read and dump unigram and bigram dictionary
      if args.ngram_freq[-3:] != 'pkl':
         d = defaultdict(float)
         with open(args.ngram_freq, 'r') as fin:
            for line in fin:
               ss = line.strip().split()
               # word is unigram or bigram
               word = ' '.join(ss[:-1])
               freq = float(ss[-1])
               d[word] = freq
         with open('dat/misc/msngram.pkl', 'wb') as fout:
            cPickle.dump(d, fout)
      else:
         with open(args.ngram_freq, 'rb') as fin:
            d = cPickle.load(fin)
      word_rank = d

   src_lang = args.src
   tgt_lang = args.tgt
   mt_para_corpus = ParallelCorpus(args.src_file, src_lang, args.mt_file, tgt_lang, args.mt_align_file)
   si_para_corpus = ParallelCorpus(args.src_file, src_lang, args.si_file, tgt_lang, args.si_align_file)
   filter_by_alignments(mt_para_corpus, si_para_corpus)

   if args.invert:
      print '======== inversion ========'
      intersect_alignments(mt_para_corpus, si_para_corpus)
      compare_inversions(mt_para_corpus, si_para_corpus, u'動詞', 'verb_inv.txt')
      compare_inversions(mt_para_corpus, si_para_corpus, u'名詞', 'verb_inv.txt')
      compare_inversions(mt_para_corpus, si_para_corpus)

   if args.omission:
      print '======== omission ========'
      compare_omission(mt_para_corpus, si_para_corpus, 'src')
      print '======== insertion ========'
      compare_omission(mt_para_corpus, si_para_corpus, 'tgt')

   if args.word_rank:
      assert word_rank is not None
      print '======== word rank ========'
      compare_word_rank(mt_para_corpus, si_para_corpus, word_rank)

   # get aligned words
   if args.vocab:
      print '======== vocabulary ========'
      vocab = Vocabulary()
      vocab.vocabulary(mt_para_corpus, si_para_corpus)

   # bleu score
   if args.bleu_N:
      print '======== bleu score ========'
      score = bleu_score(mt_para_corpus, si_para_corpus, N=args.bleu_N)
      print 'blue score (ref: mt):', score
      score = bleu_score(si_para_corpus, mt_para_corpus, N=args.bleu_N)
      print 'blue score (ref: si):', score

   if args.vw:
      print '======== classification ========'
      data = args.src[0] + args.tgt[0]
      # stat, tag and lex features, Feat defined in lib/vw.py
      for feat in range(3):
        print '======== feat %d ========' % feat
        ftrain = '%s/%s.f%d.train' % (scratch, data, feat)
        ftest = '%s/%s.f%d.test' % (scratch, data, feat)
        fmodel = '%s/%s.f%d.model' % (scratch, data, feat)
        feat_mask = [False] * Feat.size
        feat_mask[feat] = True
        vw = VW(fname_train=ftrain, fname_test=ftest, fname_model=fmodel, K=10)
        vw.write_train_test(mt_para_corpus, si_para_corpus, feat_mask, word_rank=word_rank)
        vw.train(l1=0.00005, npass=2, l=0.8)
        vw.test()
        vw.rank_feat('%s.%d.feat.rank' % (data, feat))

   if args.segment:
      print '======== segmentation ========'
      compare_segments(mt_para_corpus, si_para_corpus)

   if args.passive:
      print '======== passivization ========'
      data = args.src[0] + args.tgt[0]
      compare_passive(mt_para_corpus, si_para_corpus)

   if args.length:
      print '======== sentence length ========'
      compare_length(mt_para_corpus, si_para_corpus)

   if args.stat:
      print '======== corpus stats ========'
      print "# of pairs:", len(mt_para_corpus.sent_pairs)
      print "# of inter tokens", mean([len(sent_pair.tgt_sent.words) for sent_pair in si_para_corpus.sent_pairs])
      print "# of trans tokens", mean([len(sent_pair.tgt_sent.words) for sent_pair in mt_para_corpus.sent_pairs])

