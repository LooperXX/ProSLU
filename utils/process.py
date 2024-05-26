import time
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils.config import *
from tqdm import tqdm
from transformers import BertTokenizer, XLNetTokenizer, ElectraTokenizer, AdamW
from utils import miulab


class Processor(object):
    def __init__(self, dataset, model, args):
        self.dataset = dataset
        self.model = model
        self.batch_size = args.batch_size
        self.load_dir = args.load_dir
        self.args = args

        if self.args.gpu:
            time_start = time.time()
            self.model = self.model.cuda()
            time_con = time.time() - time_start
            mylogger.info("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con))

        self.criterion = nn.NLLLoss()
        if self.args.use_pretrained:

            if self.args.model_type == 'BERT' or self.args.model_type == 'RoBERTa':
                self.tokenizer = BertTokenizer.from_pretrained(args.model_type_path)
            elif self.args.model_type == 'XLNet':
                self.tokenizer = XLNetTokenizer.from_pretrained(args.model_type_path)
            elif self.args.model_type == 'ELECTRA':
                self.tokenizer = ElectraTokenizer.from_pretrained(args.model_type_path)
            elif self.args.model_type == 'ALBERT':
                self.tokenizer = BertTokenizer.from_pretrained(args.model_type_path)

            bert = list(map(id, self.model.encoder.parameters()))
            base_params = filter(lambda p: id(p) not in bert, self.model.parameters())
            self.optimizer = optim.Adam(
                base_params, lr=self.args.learning_rate,
                weight_decay=self.args.l2_penalty
            )

            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0},
                {'params': [p for n, p in self.model.encoder.named_parameters() if
                            not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01}
            ]
            self.optimizer_bert = AdamW(optimizer_grouped_parameters, lr=args.bert_learning_rate)

        else:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.l2_penalty
            )

        if self.load_dir:
            if self.args.gpu:
                mylogger.info("MODEL {} LOADED".format(str(self.load_dir)))
                self.model = torch.load(os.path.join(self.load_dir, 'model.pkl'))
            else:
                mylogger.info("MODEL {} LOADED".format(str(self.load_dir)))
                self.model = torch.load(os.path.join(self.load_dir, 'model.pkl'), map_location=torch.device('cpu'))

    def tokenize_batch(self, word_batch):
        piece_batch = []
        piece_span = []

        for sent_i in range(0, len(word_batch)):
            piece_batch.append([self.tokenizer.cls_token_id])
            piece_span.append([])
            count = 0

            for word_i in range(0, len(word_batch[sent_i])):
                word = word_batch[sent_i][word_i]
                piece_list = self.tokenizer.convert_tokens_to_ids([word])
                piece_batch[-1].extend(piece_list)
                piece_span[-1].append(count)
                count += len(piece_list)

            piece_batch[-1].append(self.tokenizer.sep_token_id)
        return piece_batch, piece_span

    def padding_index(self, data):
        len_list = [len(piece) for piece in data]
        max_len = max(len_list)
        for index in range(len(data)):
            data[index].extend([self.tokenizer.pad_token_id] * (max_len - len_list[index]))
        return data

    def padding_text(self, data):
        data, span = self.tokenize_batch(data)
        data = self.padding_index(data)
        return data, span

    def trans_up_ca(self, sorted_up, sorted_ca):
        up_var, ca_var = [], []
        for item_up, item_ca in zip(sorted_up, sorted_ca):
            up, ca = [], []
            up.extend(item_up)
            ca.extend(item_ca)
            up = list(Evaluator.expand_list(up))
            ca = list(Evaluator.expand_list(ca))
            up_var.append(up)
            ca_var.append(ca)
        up_var = self._cuda(torch.FloatTensor(up_var))
        ca_var = self._cuda(torch.FloatTensor(ca_var))
        return up_var, ca_var

    def _cuda(self, x):
        if self.args.gpu:
            return x.cuda()
        else:
            return x

    def train(self):
        best_dev_sent = 0.0
        no_improve, step = 0, 0
        dataloader = self.dataset.batch_delivery('train')
        total_slot_loss, total_intent_loss = 0.0, 0.0
        for epoch in range(0, self.dataset.num_epoch):

            time_start = time.time()
            self.model.train()

            for text_batch, slot_batch, intent_batch, kg_batch, up_batch, ca_batch in tqdm(dataloader, ncols=50):
                padded_text, padded_kg, [sorted_slot, sorted_intent, sorted_up, sorted_ca], \
                seq_lens, kg_lens, kg_count = self.dataset.add_padding(text_batch, kg_batch,
                                                                       [(slot_batch, True),
                                                                        (intent_batch, False),
                                                                        (up_batch, False),
                                                                        (ca_batch, False)],
                                                                       use_pretrained=self.args.use_pretrained,
                                                                       split_index=
                                                                       self.dataset.word_alphabet.get_index(['；'])[0],
                                                                       max_length=self.args.max_length)
                if self.args.use_pretrained:
                    padded_text, span = self.padding_text(padded_text)
                text_var = self._cuda(torch.LongTensor(padded_text))
                if self.args.use_pretrained:
                    text_var = (text_var, span)
                kg_var = self._cuda(torch.LongTensor(padded_kg))
                slot_var = self._cuda(torch.LongTensor(sorted_slot))
                intent_var = self._cuda(torch.LongTensor(sorted_intent)).squeeze(-1)
                up_var, ca_var = self.trans_up_ca(sorted_up, sorted_ca)

                # batch input
                random_slot = random.random()
                if random_slot < self.dataset.slot_forcing_rate:
                    slot_out, intent_out = self.model(
                        text_var, seq_lens, kg_var, kg_lens, kg_count, up_var, ca_var, forced_slot=slot_var
                    )
                else:
                    slot_out, intent_out = self.model(text_var, seq_lens, kg_var, kg_lens, kg_count, up_var, ca_var)

                # flatten output
                slot_var = torch.cat([slot_var[i][:seq_lens[i]] for i in range(0, len(seq_lens))], dim=0)
                slot_loss = self.criterion(slot_out, slot_var)
                intent_loss = self.criterion(intent_out, intent_var)
                batch_loss = slot_loss + intent_loss

                try:
                    total_slot_loss += slot_loss.cpu().item()
                    total_intent_loss += intent_loss.cpu().item()
                except AttributeError:
                    total_slot_loss += slot_loss.cpu().data.numpy()[0]
                    total_intent_loss += intent_loss.cpu().data.numpy()[0]

                step += 1
                if step % self.args.logging_steps == 0:
                    fitlog.add_loss((total_slot_loss + total_intent_loss) / step, name='loss', step=step)
                    fitlog.add_loss(total_slot_loss / step, name='slot_loss', step=step)
                    fitlog.add_loss(total_intent_loss / step, name='intent_loss', step=step)

                self.optimizer.zero_grad()
                if self.args.use_pretrained:
                    self.optimizer_bert.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                if self.args.use_pretrained:
                    self.optimizer_bert.step()

            time_con = time.time() - time_start
            mylogger.info('[Epoch {:2d}]: The total slot loss on train data is {:2.6f}, intent loss is {:2.6f}, cost ' \
                          'about {:2.6f} seconds.'.format(epoch, total_slot_loss, total_intent_loss, time_con))

            time_start = time.time()
            dev_f1_score, dev_acc, dev_sent_acc = self.estimate(if_dev=True)
            fitlog.add_metric({"dev": {
                "dev f1 score": dev_f1_score,
                "dev acc": dev_acc,
                "dev sent acc": dev_sent_acc,
            }}, step=epoch + 1)

            if dev_sent_acc > best_dev_sent:
                fitlog.add_best_metric({"dev": {
                    "dev f1 score": dev_f1_score,
                    "dev acc": dev_acc,
                    "dev sent acc": dev_sent_acc,
                }})

                no_improve = 0
                best_dev_sent = dev_sent_acc
                test_f1, test_acc, test_sent_acc = self.estimate(if_dev=False)
                fitlog.add_metric({"test": {
                    "test f1 score": test_f1,
                    "test acc": test_acc,
                    "test sent acc": test_sent_acc,
                }}, step=epoch + 1)

                fitlog.add_best_metric({"test": {
                    "test f1 score": test_f1,
                    "test acc": test_acc,
                    "test sent acc": test_sent_acc,
                }})

                mylogger.info('\nTest result: slot f1 score: {:.6f}, intent acc score: {:.6f}, semantic '
                              'accuracy score: {:.6f}.'.format(test_f1, test_acc, test_sent_acc))

                model_save_dir = os.path.join(self.dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)

                torch.save(self.model, os.path.join(model_save_dir, "model.pkl"))
                torch.save(self.dataset, os.path.join(model_save_dir, 'dataset.pkl'))

                time_con = time.time() - time_start
                mylogger.info('[Epoch {:2d}]: In validation process, the slot f1 score is {:2.6f}, ' \
                              'the intent acc is {:2.6f}, the semantic acc is {:.2f}, cost about ' \
                              '{:2.6f} seconds.\n'.format(epoch, dev_f1_score, dev_acc, dev_sent_acc, time_con))
            else:
                no_improve += 1

            if self.args.early_stop:
                if no_improve > self.args.patience:
                    mylogger.info('early stop at epoch {}'.format(epoch))
                    break

    def estimate(self, if_dev=True):
        """
        Estimate the performance of model on dev or test dataset.
        """
        with torch.no_grad():
            if if_dev:
                pred_slot, real_slot, pred_intent, real_intent, _ = self.prediction("dev")
            else:
                pred_slot, real_slot, pred_intent, real_intent, _ = self.prediction("test")

        slot_f1_socre = miulab.computeF1Score(real_slot, pred_slot)[0]
        intent_acc = Evaluator.accuracy(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)

        return slot_f1_socre, intent_acc, sent_acc

    def validate(self, model_path, dataset_path):
        """
        validation will write mistaken samples to files and make scores.
        """

        if self.args.gpu:
            self.model = torch.load(model_path)
            self.dataset = torch.load(dataset_path)
        else:
            self.model = torch.load(model_path, map_location=torch.device('cpu'))
            self.dataset = torch.load(dataset_path, map_location=torch.device('cpu'))

        self.dataset.quick_build_test(self.args.data_dir, 'test.json')
        mylogger.info('load {} to test'.format(self.args.data_dir))

        # Get the sentence list in test dataset.
        # sent_list = dataset.test_sentence
        with torch.no_grad():
            pred_slot, real_slot, pred_intent, real_intent, sent_list = self.prediction("test")

        slot_f1 = miulab.computeF1Score(real_slot, pred_slot)[0]
        intent_acc = Evaluator.accuracy(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)

        # To make sure the directory for save error prediction.
        mistake_dir = os.path.join(self.dataset.save_dir, "error")
        if not os.path.exists(mistake_dir):
            os.mkdir(mistake_dir)

        slot_file_path = os.path.join(mistake_dir, "slot.txt")
        intent_file_path = os.path.join(mistake_dir, "intent.txt")
        both_file_path = os.path.join(mistake_dir, "both.txt")

        # Write those sample with mistaken slot prediction.
        with open(slot_file_path, 'w') as fw:
            for w_list, r_slot_list, p_slot_list in zip(sent_list, real_slot, pred_slot):
                if r_slot_list != p_slot_list:
                    for w, r, p in zip(w_list, r_slot_list, p_slot_list):
                        fw.write(w + '\t' + r + '\t' + p + '\n')
                    fw.write('\n')

        # Write those sample with mistaken intent prediction.
        with open(intent_file_path, 'w') as fw:
            for w_list, p_intent, r_intent in zip(sent_list, pred_intent, real_intent):
                if p_intent != r_intent:
                    for w in w_list:
                        fw.write(w + '\n')
                    fw.write(r_intent + '\t' + p_intent + '\n\n')

        # Write those sample both have intent and slot errors.
        with open(both_file_path, 'w') as fw:
            for w_list, r_slot_list, p_slot_list, p_intent, r_intent in \
                    zip(sent_list, real_slot, pred_slot, pred_intent, real_intent):

                if r_slot_list != p_slot_list or r_intent != p_intent:
                    for w, r_slot, p_slot in zip(w_list, r_slot_list, p_slot_list):
                        fw.write(w + '\t' + r_slot + '\t' + p_slot + '\n')
                    fw.write(r_intent + '\t' + p_intent + '\n\n')

        return slot_f1, intent_acc, sent_acc

    def prediction(self, mode):
        self.model.eval()

        if mode == "dev":
            dataloader = self.dataset.batch_delivery('dev', batch_size=self.args.batch_size,
                                                     shuffle=False, is_digital=False)
        elif mode == "test":
            dataloader = self.dataset.batch_delivery('test', batch_size=self.args.batch_size,
                                                     shuffle=False, is_digital=False)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")

        pred_slot, real_slot = [], []
        pred_intent, real_intent = [], []
        sent_list = []

        for text_batch, slot_batch, intent_batch, kg_batch, up_batch, ca_batch in tqdm(dataloader, ncols=50):
            padded_text, padded_kg, [sorted_slot, sorted_intent, sorted_up, sorted_ca], \
            seq_lens, kg_lens, kg_count = self.dataset.add_padding(text_batch, kg_batch,
                                                                   [(slot_batch, False),
                                                                    (intent_batch, False),
                                                                    (up_batch, False),
                                                                    (ca_batch, False)],
                                                                   digital=False,
                                                                   use_pretrained=self.args.use_pretrained,
                                                                   split_index=
                                                                   self.dataset.word_alphabet.get_index(['；'])[0],
                                                                   max_length=self.args.max_length)
            real_slot.extend(sorted_slot)
            real_intent.extend(list(Evaluator.expand_list(sorted_intent)))

            if self.args.use_pretrained:
                sent_list.extend(padded_text)
                padded_text, span = self.padding_text(padded_text)
                var_text = self._cuda(torch.LongTensor(padded_text))
                var_text = (var_text, span)
            else:
                sent_list.extend([pt[:seq_lens[idx]] for idx, pt in enumerate(padded_text)])
                digit_text = self.dataset.word_alphabet.get_index(padded_text)
                var_text = self._cuda(torch.LongTensor(digit_text))

            digit_kg = self.dataset.word_alphabet.get_index(padded_kg)
            var_kg = self._cuda(torch.LongTensor(digit_kg))
            var_up, var_ca = self.trans_up_ca(sorted_up, sorted_ca)

            slot_idx, intent_idx = self.model(var_text, seq_lens, var_kg, kg_lens, kg_count, var_up, var_ca,
                                              n_predicts=1)
            nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], seq_lens)[0]
            pred_slot.extend(self.dataset.slot_alphabet.get_instance(nested_slot))
            pred_intent.extend(self.dataset.intent_alphabet.get_instance(intent_idx))
        pred_intent = [pred_intent_[0] for pred_intent_ in pred_intent]
        return pred_slot, real_slot, pred_intent, real_intent, sent_list


class Evaluator(object):

    @staticmethod
    def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """

        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):

            if p_slot == r_slot and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """

        pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items
