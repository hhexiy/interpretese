import random, codecs
#from word_alignment import *
from corpus import *
from subprocess import Popen, PIPE
from collections import defaultdict

class Feat:
   stat = 0
   tag = 1
   lex = 2
   lextag = 3
   verb = 4
   marker = 5
   ctag = 6
   size = 7

class VW:
   def __init__(self, fname_train, fname_test, fname_model, K=10):
      self.fname_train_k = ['%s.%d' % (fname_train, i) for i in range(K)]
      self.fname_test_k = ['%s.%d' % (fname_test, i) for i in range(K)]
      self.fname_pred_k = ['%s.pred.%d' % (fname_test, i) for i in range(K)]
      self.fname_model_k = ['%s.%d' % (fname_model, i) for i in range(K)]
      self.fname_hash_k = ['%s.hash' % (fname_model) for fname_model in self.fname_model_k]
      self.K = K   # K-fold cross validation

   def train(self, l1=0, l2=0, npass=1, loss='logistic', l=0.5):
      print 'npass:', npass, 'learning rate:', l
      for i in range(self.K):
         fname_train = self.fname_train_k[i]
         fname_model = self.fname_model_k[i]
         args = 'vw --quiet -d %s --loss_function %s -f %s --readable_model %s --l1 %f --l2 %f --passes %d -c -k -l %f' % (fname_train, loss, fname_model, fname_model+'.txt', l1, l2, npass, l)
         p = Popen(args.split(), stdout=PIPE)
         ret = p.communicate()

   def test(self):
      #acc = []
      for i in range(self.K):
         fname_test = self.fname_test_k[i]
         fname_model = self.fname_model_k[i]
         fname_hash = self.fname_hash_k[i]
         fname_pred = self.fname_pred_k[i]
         args = 'vw --quiet -i %s -t %s -p %s --invert_hash %s' % (fname_model, fname_test, fname_pred, fname_hash)
         p = Popen(args.split(), stdout=PIPE)
         p.communicate()

      precision, recall, fscore, accuracy = self.get_acc()
      print 'precision:', precision
      print 'recall:', recall
      print 'fscore:', fscore
      print 'accuracy:', accuracy

   def rank_feat(self, fname):
      feat_dict, feat_weights = self.build_feat_dict()
      sorted_feat = sorted(feat_weights.items(), key=lambda x: abs(x[1]), reverse=True)
      with codecs.open(fname, 'w') as fout:
         for hashcode, weight in sorted_feat:
            fout.write('%s\t%f\n' % (feat_dict[hashcode], weight))

   def build_feat_dict(self):
      feat_dict = {}
      feat_weights = defaultdict(list)
      for fname_hash in self.fname_hash_k:
         with open(fname_hash, 'r') as fin:
            num_lines_header = 12
            count = 0
            for line in fin:
               if count < 12:
                  count += 1
                  continue
               # name:hashcode:value
               name, hashcode, weight = line.strip().split(':')
               weight = float(weight)
               # NOTE: there are collisons!
               #if hashcode in feat_dict:
               #   assert feat_dict[hashcode] == name
               #else:
               #   feat_dict[hashcode] = name
               feat_dict[hashcode] = name
               feat_weights[hashcode].append(weight)
      # average weights
      for feat, weights in feat_weights.items():
         feat_weights[feat] = sum(weights) / float(len(weights))
      return feat_dict, feat_weights

   def get_acc(self):
      for i in range(self.K):
         fname_ref = self.fname_test_k[i]
         fname_pred = self.fname_pred_k[i]
         pred = []
         ref = []
         with open(fname_ref, 'r') as fref, open(fname_pred, 'r') as fpred:
            for line_ref , line_pred in zip(fref, fpred):
               ref.append(int(line_ref.strip().split(' ', 1)[0]))
               pred.append(1 if float(line_pred.strip()) > 0 else -1)

      # f-score
      # true positive
      tp = sum([1 if p == 1 and r == 1 else 0 for p, r in zip(pred, ref)])
      # predicted positive
      pp = sum([1 if p == 1 else 0 for p in pred])
      # positive examples
      rp = sum([1 if r == 1 else 0 for r in ref])
      precision = float(tp) / pp
      recall = float(tp) / rp
      fscore = 2 * (precision*recall) / (precision+recall)
      ncorrect = sum([1 if p == r else 0 for p, r in zip(pred, ref)])
      acc = ncorrect / float(len(pred))
      return precision, recall, fscore, acc

   def write_train_test_subset(self, sents, inst_masks, labels, weights, feat_mask, upto=-1):
      '''
      write vw examples for a subset of instances (True in inst_mask).
      inst_mask and label should have the same dimension as sent_pairs in para_corpus.
      extract features (True in feat_mask) from sentences in sents (list of lists)
      '''
      folds = []
      examples = []
      assert len(sents) == len(labels)
      i = 0
      for inst_mask, label, weight, sent_list in zip(inst_masks, labels, weights, sents):
         if not inst_mask:
            continue
         # one example for one sent_list
         fold = i % self.K
         folds.append(fold)
         i += 1
         # sent_list can be single src sent of src sent and mt sent
         feat_list = []
         for sent in sent_list:
            feat_list.append(sent.get_vw_feat(feat_mask, upto=upto))
         feat_str = ' '.join(feat_list)
         examples.append('%d %f %s' % (label, weight, feat_str))

      #random.shuffle(folds)

      # write
      for fold in range(self.K):
         fname_train = self.fname_train_k[fold]
         fname_test = self.fname_test_k[fold]
         with codecs.open(fname_train, 'w', encoding='utf-8' ) as ftrain, codecs.open(fname_test, 'w', encoding='utf-8' ) as ftest:
            for idx, example in zip(folds, examples):
               if idx == fold:
                  fout = ftest
               else:
                  fout = ftrain
               fout.write('%s\n' % example);

   # TODO: integrate in the same interface as _subset
   def write_train_test(self, mt_para_corpus, si_para_corpus, feat_mask, word_rank=None):
      for para_corpus in [mt_para_corpus, si_para_corpus]:
         for sent_pair in para_corpus.sent_pairs:
            sent_pair.compute_bilingual_feats()
      # get feature and fold assignment
      folds = []
      feats = []
      i = 0
      for si_sent_pair, mt_sent_pair in zip(si_para_corpus.sent_pairs, mt_para_corpus.sent_pairs):
         if si_sent_pair.good_alignment and mt_sent_pair.good_alignment:
            fold = i % self.K
            folds.append(fold)
            i += 1
            si_sent_feat = si_sent_pair.tgt_sent.get_vw_feat(feat_mask, word_rank)
            mt_sent_feat = mt_sent_pair.tgt_sent.get_vw_feat(feat_mask, word_rank)
            feats.append((si_sent_feat, mt_sent_feat))

      #random.shuffle(folds)

      # write
      for fold in range(self.K):
         fname_train = self.fname_train_k[fold]
         fname_test = self.fname_test_k[fold]
         with codecs.open(fname_train, 'w', encoding='utf-8' ) as ftrain, codecs.open(fname_test, 'w', encoding='utf-8' ) as ftest:
            for idx, (si_sent_feat, mt_sent_feat) in zip(folds, feats):
               if idx == fold:
                  fout = ftest
               else:
                  fout = ftrain
               fout.write('1 %s\n' % si_sent_feat);
               fout.write('-1 %s\n' % mt_sent_feat);




