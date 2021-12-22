import os
import json
import numpy as np
from copy import deepcopy
from collections import Counter
from collections import OrderedDict
from ordered_set import OrderedSet

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.config import *


class Alphabet(object):
    """
    Storage and serialization a set of elements.
    """

    def __init__(self, name, if_use_pad, if_use_unk):

        self.name = name
        self.if_use_pad = if_use_pad
        self.if_use_unk = if_use_unk

        self.index2instance = OrderedSet()
        self.instance2index = OrderedDict()

        # Counter Object record the frequency
        # of element occurs in raw text.
        self.counter = Counter()

        if if_use_pad:
            self.sign_pad = "<PAD>"
            self.add_instance(self.sign_pad)
        if if_use_unk:
            self.sign_unk = "<UNK>"
            self.add_instance(self.sign_unk)

    def index2instance(self):
        return self.index2instance

    def add_instance(self, instance):
        """ Add instances to alphabet.

        1, We support any iterative data structure which
        contains elements of str type.

        2, We will count added instances that will influence
        the serialization of unknown instance.

        :param instance: is given instance or a list of it.
        """

        if isinstance(instance, (list, tuple)):
            for element in instance:
                self.add_instance(element)
            return

        # We only support elements of str type.
        assert isinstance(instance, str)

        # count the frequency of instances.
        self.counter[instance] += 1

        if instance not in self.index2instance:
            self.instance2index[instance] = len(self.index2instance)
            self.index2instance.append(instance)

    def get_index(self, instance):
        """ Serialize given instance and return.

        For unknown words, the return index of alphabet
        depends on variable self.use_unk:

            1, If True, then return the index of "<UNK>";
            2, If False, then return the index of the
            element that hold max frequency in training data.

        :param instance: is given instance or a list of it.
        :return: is the serialization of query instance.
        """

        if isinstance(instance, (list, tuple)):
            return [self.get_index(elem) for elem in instance]

        assert isinstance(instance, str)

        try:
            return self.instance2index[instance]
        except KeyError:
            if self.if_use_unk:
                return self.instance2index[self.sign_unk]
            else:
                max_freq_item = self.counter.most_common(1)[0][0]
                return self.instance2index[max_freq_item]

    def get_instance(self, index):
        """ Get corresponding instance of query index.

        if index is invalid, then throws exception.

        :param index: is query index, possibly iterable.
        :return: is corresponding instance.
        """

        if isinstance(index, list):
            return [self.get_instance(elem) for elem in index]

        return self.index2instance[index]

    def save_content(self, dir_path):
        """ Save the content of alphabet to files.

        There are two kinds of saved files:
            1, The first is a list file, elements are
            sorted by the frequency of occurrence.

            2, The second is a dictionary file, elements
            are sorted by it serialized index.

        :param dir_path: is the directory path to save object.
        """

        # Check if dir_path exists.
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        list_path = os.path.join(dir_path, self.name + "_list.txt")
        with open(list_path, 'w') as fw:
            for element, frequency in self.counter.most_common():
                fw.write(element + '\t' + str(frequency) + '\n')

        dict_path = os.path.join(dir_path, self.name + "_dict.txt")
        with open(dict_path, 'w') as fw:
            for index, element in enumerate(self.index2instance):
                fw.write(element + '\t' + str(index) + '\n')

    def __len__(self):
        return len(self.index2instance)

    def __str__(self):
        return 'Alphabet {} contains about {} words: \n\t{}'.format(self.name, len(self), self.index2instance)


class TorchDataset(Dataset):
    """
    Helper class implementing torch.utils.data.Dataset to
    instantiate DataLoader which deliveries data batch.
    """

    def __init__(self, text, slot, intent, kg, up, ca):
        self.text = text
        self.slot = slot
        self.intent = intent
        self.kg = kg
        self.up = up
        self.ca = ca

    def __getitem__(self, index):
        return self.text[index], self.slot[index], self.intent[index], self.kg[index], \
               [u[index] for u in self.up], \
               [c[index] for c in self.ca]

    def __len__(self):
        # Pre-check to avoid bug.
        assert len(self.text) == len(self.slot)
        assert len(self.text) == len(self.intent)
        assert len(self.text) == len(self.kg)
        assert len(self.text) == len(self.up[0])
        assert len(self.text) == len(self.ca[0])

        return len(self.text)


class DatasetManager(object):

    def __init__(self, args):

        # Instantiate alphabet objects.
        self.word_alphabet = Alphabet('word', if_use_pad=True, if_use_unk=True)
        self.slot_alphabet = Alphabet('slot', if_use_pad=False, if_use_unk=False)
        self.intent_alphabet = Alphabet('intent', if_use_pad=False, if_use_unk=False)

        # Record the raw text of dataset.
        self.text_word_data = {}
        self.text_kg_data = {}
        self.text_slot_data = {}
        self.text_intent_data = {}

        # Record the serialization of dataset.
        self.digit_word_data = {}
        self.digit_kg_data = {}
        self.digit_slot_data = {}
        self.digit_intent_data = {}

        self.digit_up_data = {}
        self.digit_ca_data = {}

        self.args = args

    @property
    def test_sentence(self):
        return deepcopy(self.text_word_data['test'])

    @property
    def num_epoch(self):
        return self.args.num_epoch

    @property
    def batch_size(self):
        return self.args.batch_size

    @property
    def learning_rate(self):
        return self.args.learning_rate

    @property
    def l2_penalty(self):
        return self.args.l2_penalty

    @property
    def save_dir(self):
        return self.args.save_dir

    @property
    def slot_forcing_rate(self):
        return self.args.slot_forcing_rate

    def quick_build(self):
        """
        Convenient function to instantiate a dataset object.
        """

        train_path = os.path.join(self.args.data_dir, 'train.json')
        dev_path = os.path.join(self.args.data_dir, 'dev.json')
        test_path = os.path.join(self.args.data_dir, 'test.json')

        self.add_file(train_path, 'train', train_file=True)
        self.add_file(dev_path, 'dev', train_file=False)
        self.add_file(test_path, 'test', train_file=False)

        # Check if save path exists.
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        alphabet_dir = os.path.join(self.save_dir, "alphabet")
        self.word_alphabet.save_content(alphabet_dir)
        self.slot_alphabet.save_content(alphabet_dir)
        self.intent_alphabet.save_content(alphabet_dir)

    def quick_build_test(self, data_dir, file_name):
        """
        Convenient function to instantiate a dataset object.
        """

        test_path = os.path.join(data_dir, file_name)
        self.add_file(test_path, 'test', train_file=False)

    def add_file(self, file_path, data_name, train_file):
        kg, up, ca, text, slot, intent = self.read_info_file(file_path)

        if train_file:
            self.word_alphabet.add_instance(text)
            self.word_alphabet.add_instance(kg)
            self.slot_alphabet.add_instance(slot)
            self.intent_alphabet.add_instance(intent)

        # Record the raw text of dataset.
        self.text_word_data[data_name] = text
        self.text_kg_data[data_name] = kg
        self.text_slot_data[data_name] = slot
        self.text_intent_data[data_name] = intent

        # Serialize raw text and stored it.
        self.digit_word_data[data_name] = self.word_alphabet.get_index(text)
        self.digit_kg_data[data_name] = self.word_alphabet.get_index(kg)
        self.digit_up_data[data_name] = up
        self.digit_ca_data[data_name] = ca
        if train_file:
            self.digit_slot_data[data_name] = self.slot_alphabet.get_index(slot)
            self.digit_intent_data[data_name] = self.intent_alphabet.get_index(intent)

    def read_info_file(self, file_path):
        with open(file_path, 'r') as fr:
            data = json.load(fr)
            ids = data['ids']
            len_up, len_ca = [3, 3, 3, 2], [8, 3, 4, 3]  # Manual Setting
            kgs, ups, cas = [], [], []
            texts, slots, intents = [], [], []

            for i in range(len(len_up)):
                ups.append([])
            for i in range(len(len_ca)):
                cas.append([])

            for id in ids:
                kgs.append([i for i in data[id]['KG']])
                texts.append(list(data[id]['用户话语']))
                slots.append(data[id]['slot'])
                intents.append([data[id]['intent']])
                if kgs[-1] == []:
                    kgs[-1] = [' ']
                if data[id]['UP'] != []:
                    index = 0
                    for key in self.args.up_keys:
                        ups[index].append(data[id]['UP'][key])
                        index += 1
                else:
                    index = 0
                    for key in self.args.up_keys:
                        ups[index].append([0.0] * len_up[index])
                        index += 1
                if data[id]['CA'] != []:
                    index = 0
                    for key in self.args.ca_keys:
                        cas[index].append(data[id]['CA'][key])
                        index += 1
                else:
                    index = 0
                    for key in self.args.ca_keys:
                        cas[index].append([0.0] * len_ca[index])
                        index += 1
        return kgs, ups, cas, texts, slots, intents

    def batch_delivery(self, data_name, batch_size=None, is_digital=True, shuffle=True):
        if batch_size is None:
            batch_size = self.batch_size

        if is_digital:
            if self.args.use_pretrained:
                text = self.text_word_data[data_name]
            else:
                text = self.digit_word_data[data_name]
            kg = self.digit_kg_data[data_name]
            slot = self.digit_slot_data[data_name]
            intent = self.digit_intent_data[data_name]
        else:
            text = self.text_word_data[data_name]
            kg = self.text_kg_data[data_name]
            slot = self.text_slot_data[data_name]
            intent = self.text_intent_data[data_name]

        up = self.digit_up_data[data_name]
        ca = self.digit_ca_data[data_name]
        dataset = TorchDataset(text, slot, intent, kg, up, ca)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    @staticmethod
    def add_padding(texts, kgs, items, digital=True, use_pretrained=False, split_index=None, max_length=None):
        len_list = [len(text) for text in texts]
        max_len = max(len_list)
        # Get sorted index of len_list.
        sorted_index = np.argsort(len_list)[::-1]

        trans_texts, seq_lens, trans_items = [], [], None
        trans_kg, kg_lens, kg_count = [], [], []
        trans_items = [[] for _ in range(0, len(items))]

        for index in sorted_index:
            seq_lens.append(deepcopy(len_list[index]))
            trans_texts.append(deepcopy(texts[index]))
            trans_kg.append(deepcopy(kgs[index]))

            if not use_pretrained:
                if digital:
                    trans_texts[-1].extend([0] * (max_len - len_list[index]))
                else:
                    trans_texts[-1].extend(['<PAD>'] * (max_len - len_list[index]))

            for item, (o_item, required) in zip(trans_items, items):
                item.append(deepcopy(o_item[index]))
                if required:
                    if digital:
                        item[-1].extend([0] * (max_len - len_list[index]))
                    else:
                        item[-1].extend(['<PAD>'] * (max_len - len_list[index]))

        kg_split, kg_count = [], []
        for kg in trans_kg:
            ids = []
            while split_index in kg:
                split = kg.index(split_index)
                ids.append(kg[:min(max_length, split)])
                kg = kg[split + 1:]
                if ids[-1] == []:
                    ids.pop(-1)
            if kg != []:
                ids.append(kg[:max_length])
            kg_split.extend(ids)
            kg_count.append(len(ids))
        trans_kg = kg_split
        kg_lens = [len(kg) for kg in trans_kg]
        max_len_kg = max(kg_lens)
        for i in range(len(trans_kg)):
            if digital:
                trans_kg[i].extend([0] * (max_len_kg - kg_lens[i]))
            else:
                trans_kg[i].extend(['<PAD>'] * (max_len_kg - kg_lens[i]))

        return trans_texts, trans_kg, trans_items, seq_lens, kg_lens, kg_count

    @staticmethod
    def collate_fn(batch):
        """
        helper function to instantiate a DataLoader Object.
        """

        n_entity = len(batch[0])
        modified_batch = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(batch)):
            for jdx in range(0, n_entity):
                modified_batch[jdx].append(batch[idx][jdx])

        return modified_batch
