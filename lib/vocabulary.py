import numpy as np
import codecs
from scipy import stats
from util import ftest, ttest

class Vocabulary:

   def vocabulary(self, mt_para_corpus, si_para_corpus):
      mt_vocab = mt_para_corpus.get_tgt_vocab('mt.vocab.txt')
      si_vocab = si_para_corpus.get_tgt_vocab('si.vocab.txt')
      print 'mt vocab mean and std:', np.mean(mt_vocab.values()), np.std(mt_vocab.values())
      print 'si vocab mean and std:', np.mean(si_vocab.values()), np.std(si_vocab.values())

      mt_vocab_set = set(mt_vocab.keys())
      si_vocab_set = set(si_vocab.keys())
      mt_si_diff = mt_vocab_set - si_vocab_set
      si_mt_diff = si_vocab_set - mt_vocab_set
      mt_and_si_vocab = {}
      for ja_tok in  mt_vocab_set & si_vocab_set:
         mt_and_si_vocab[ja_tok] = (mt_vocab[ja_tok], si_vocab[ja_tok])
      with codecs.open('mt_si_vocab_diff.txt', 'w', encoding='utf-8') as fout:
         mt_and_si_vocab_diff = sorted(mt_and_si_vocab.items(), key=lambda x: x[1][0] - x[1][1], reverse=True)
         for ja_tok, (mt_freq, si_freq) in mt_and_si_vocab_diff:
            fout.write('%s\t%d %d\n' % (ja_tok, mt_freq, si_freq))

      #with codecs.open('mt-si.vocab.txt', 'w', encoding='utf-8') as fout:
      #   for tok in mt_si_diff:
      #      fout.write('%s\n' % tok)
      #with codecs.open('si-mt.vocab.txt', 'w', encoding='utf-8') as fout:
      #   for tok in si_mt_diff:
      #      fout.write('%s\n' % tok)

   def print_trans(self, fname, ind, en_toks, mt_trans, si_trans):
      with codecs.open(fname, 'w', encoding='utf-8') as fout:
         for i in ind:
            fout.write('%-10s%-10s%-10s\n' % ('En tok', 'MT trans', 'SI trans'))
            fout.write('%s\n' % en_toks[i])
            ntrans = max(len(mt_trans[i]), len(si_trans[i]))
            mt_trans[i].extend([('', 0)] * (ntrans - len(mt_trans[i])))
            si_trans[i].extend([('', 0)] * (ntrans - len(si_trans[i])))
            assert len(mt_trans[i]) == len(si_trans[i])
            for mt_ja, si_ja in zip(mt_trans[i], si_trans[i]):
               fout.write('%s %d\t\t%s %d\n' % (mt_ja[0], mt_ja[1], si_ja[0], si_ja[1]))
            fout.write('\n')

