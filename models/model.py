from models.module import *


class ModelManager(nn.Module):

    def __init__(self, args, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()

        self.num_word = num_word
        self.num_slot = num_slot
        self.num_intent = num_intent
        self.args = args

        self.embedding = nn.Embedding(
            self.num_word,
            self.args.word_embedding_dim
        )

        self.encoder_kg = LSTMEncoder(
            self.args.word_embedding_dim,
            self.args.info_embedding_dim,
            self.args.dropout_rate
        )

        # Fusion Setting
        if args.use_info:
            self.hierarchical_fusion = Hierarchical_Fusion(
                self.args.encoder_hidden_dim if args.use_pretrained else self.args.encoder_hidden_dim + self.args.attention_output_dim,
                self.args.info_embedding_dim,
                self.args.dropout_rate
            )

        # Initialize an embedding object.
        if args.use_pretrained:
            # Initialize a BERT Encoder object.
            if self.args.model_type == 'BERT' or self.args.model_type == 'RoBERTa':
                self.encoder = BertEncoder(self.args.bert_dropout_rate)
            elif self.args.model_type == 'XLNet':
                self.encoder = XLNetEncoder(self.args.bert_dropout_rate)
            elif self.args.model_type == 'ELECTRA':
                self.encoder = ElectraEncoder(self.args.bert_dropout_rate)
            elif self.args.model_type == 'ALBERT':
                self.encoder = AlbertEncoder(self.args.bert_dropout_rate)

            intent_info = self.args.info_embedding_dim if self.args.use_info else 0
            self.intent_decoder = nn.Linear(
                self.args.encoder_hidden_dim + intent_info, self.num_intent
            )

            # Initialize a Decoder object for slot.
            self.slot_decoder = LSTMDecoder(
                self.args.encoder_hidden_dim,
                self.args.slot_decoder_hidden_dim,
                self.num_slot, self.args.dropout_rate,
                embedding_dim=self.args.slot_embedding_dim,
                extra_dim=self.num_intent,
                info_dim=self.args.info_embedding_dim if self.args.use_info else 0,
                args=self.args
            )

        else:
            # Initialize an LSTM Encoder object.
            self.encoder = LSTMEncoder(
                self.args.word_embedding_dim,
                self.args.encoder_hidden_dim,
                self.args.dropout_rate
            )

            # Initialize a self-attention layer.
            self.attention = SelfAttention(
                self.args.word_embedding_dim,
                self.args.attention_hidden_dim,
                self.args.attention_output_dim,
                self.args.dropout_rate
            )

            # Initialize a Decoder object for intent.
            self.sentattention = UnflatSelfAttention(
                self.args.encoder_hidden_dim + self.args.attention_output_dim,
                self.args.dropout_rate
            )

            intent_info = self.args.info_embedding_dim if self.args.use_info else 0
            self.intent_decoder = nn.Linear(
                self.args.encoder_hidden_dim + self.args.attention_output_dim + intent_info, self.num_intent
            )

            # Initialize a Decoder object for slot.
            self.slot_decoder = LSTMDecoder(
                self.args.encoder_hidden_dim + self.args.attention_output_dim,
                self.args.slot_decoder_hidden_dim,
                self.num_slot, self.args.dropout_rate,
                embedding_dim=self.args.slot_embedding_dim,
                info_dim=self.args.info_embedding_dim if self.args.use_info else 0,
                extra_dim=self.num_intent,
                args=self.args
            )

        if self.args.use_info:
            self.encoder_up = nn.Linear(
                self.args.up_input_dim,
                self.args.info_embedding_dim
            )

            self.encoder_ca = nn.Linear(
                self.args.ca_input_dim,
                self.args.info_embedding_dim
            )

        self.intent_embedding = nn.Embedding(self.num_intent, self.num_intent)
        self.intent_embedding.weight.data = torch.eye(self.num_intent)
        self.intent_embedding.weight.requires_grad = False

    def _cuda(self, x):
        if self.args.gpu:
            return x.cuda()
        else:
            return x

    def match_token(self, hiddens, span):
        # take the first subword hidden as the represenation for each token in the utterance
        hiddens_span = self._cuda(torch.zeros_like(hiddens))
        for i in range(len(span)):
            for idx, span_i in enumerate(span[i]):
                hiddens_span[i][idx] = hiddens[i][span_i]
        return hiddens_span

    def get_kg(self, hiddens, lens):
        # take the last hidden as the represenation
        embs = []
        for idx, length in enumerate(lens):
            embs.append(hiddens[idx, length - 1])
        embs = torch.stack(embs)
        return embs

    def match_kg(self, hiddens, count):
        # average the entities for each sample as KG represenation
        output = []
        index = 0
        for count_i in count:
            output.append(torch.mean(hiddens[index: index + count_i], dim=0))
            index += count_i
        output = self._cuda(torch.stack(output))
        return output

    def fusion(self, hiddens, kg_emb, up_emb, ca_emb):
        info_emb = self.hierarchical_fusion(hiddens, torch.cat([kg_emb.unsqueeze(1), up_emb.unsqueeze(1),
                                                                ca_emb.unsqueeze(1)], dim=1))
        return info_emb

    def forward(self, text, seq_lens, kg_var, kg_lens, kg_count, up_var, ca_var, n_predicts=None, forced_slot=None):
        info_emb_slot, sent_rep, info_emb_intent = None, None, None
        if self.args.use_pretrained:
            [word_tensor, span] = text
            hiddens = self.encoder(word_tensor)
            sent_rep = hiddens[:, 0]
            hiddens = hiddens[:, 1:-1]
            hiddens = self.match_token(hiddens, span)
        else:
            word_tensor = self.embedding(text)
            lstm_hiddens = self.encoder(word_tensor, seq_lens)
            attention_hiddens = self.attention(word_tensor, seq_lens)
            hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=-1)
            sent_rep = self.sentattention(hiddens, seq_lens)

        if self.args.use_info:
            kg_tensor = self.embedding(kg_var)
            kg_hiddens = self.encoder_kg(kg_tensor, kg_lens, enforce_sorted=False)
            kg_emb = self.get_kg(kg_hiddens, kg_lens)
            kg_emb = self.match_kg(kg_emb, kg_count)
            up_emb = self.encoder_up(up_var)
            ca_emb = self.encoder_ca(ca_var)
            info_emb_slot = self.fusion(hiddens, kg_emb, up_emb, ca_emb)
            info_emb_intent = self.fusion(sent_rep.unsqueeze(1), kg_emb, up_emb, ca_emb).squeeze(1)

        if self.args.use_info:
            sent_rep = torch.cat([sent_rep, info_emb_intent], dim=-1)
        pred_intent = self.intent_decoder(sent_rep)

        if not self.args.differentiable:
            _, idx_intent = pred_intent.topk(1, dim=-1)
            feed_intent = self.intent_embedding(idx_intent.squeeze(1))
        else:
            feed_intent = pred_intent

        feed_intent = feed_intent.unsqueeze(1).repeat(1, hiddens.size(1), 1)

        pred_slot = self.slot_decoder(
            hiddens, seq_lens,
            extra_input=feed_intent,
            forced_input=forced_slot,
            info_emb=info_emb_slot
        )

        if n_predicts is None:
            return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_intent, dim=1)
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            _, intent_index = pred_intent.topk(n_predicts, dim=1)
            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist()
