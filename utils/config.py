import argparse
import torch
import fitlog
import logging
import os
import datetime
import json


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt='%Y/%d/%m %H:%M:%S'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


parser = argparse.ArgumentParser()
# Dataset and Other Parameters
parser.add_argument('--data_dir', '-dd', type=str, default='data/ProSLU')
parser.add_argument('--save_dir', '-sd', type=str, default='save/')
parser.add_argument('--load_dir', '-ld', type=str, default=None)
parser.add_argument("--fix_seed", '-fs', help='fix seed', action='store_true', required=False)
parser.add_argument("--random_seed", '-rs', type=int, default=0)
parser.add_argument('--use_fitlog', '-uf', help='use fitlog', action='store_true', required=False)
parser.add_argument('--gpu', '-g', action='store_true', help='use gpu', required=False)
parser.add_argument("--use_info", '-ui', help='use info', action='store_true', required=False)
parser.add_argument('--use_pretrained', '-up', action='store_true', help='use pretrained models', required=False)
parser.add_argument('--model_type', '-mt', type=str, default="ELECTRA")
parser.add_argument('--max_length', '-ml', type=int, help='max length for KG', default=256)
parser.add_argument('--early_stop', '-es', action='store_true', required=False)
parser.add_argument('--patience', '-pa', type=int, default=10)
parser.add_argument('--fit_log_dir', '-fld', required=False, default='logs/')
parser.add_argument('--logging_steps', '-ls', type=int, default=10)

# Training parameters.
parser.add_argument('--num_epoch', '-ne', type=int, default=100)
parser.add_argument('--batch_size', '-bs', type=int, default=8)
parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.001)
parser.add_argument("--bert_learning_rate", '-blr', type=float, default=0.00001)
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.4)
parser.add_argument('--bert_dropout_rate', '-bdr', type=float, default=0.1)
parser.add_argument("--differentiable", "-d", action="store_true", default=False)
parser.add_argument('--slot_forcing_rate', '-sfr', type=float, default=0.9)

# Model parameters.
parser.add_argument('--word_embedding_dim', '-wed', type=int, default=64)
parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=768)
parser.add_argument('--slot_embedding_dim', '-sed', type=int, default=32)
parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=64)
parser.add_argument('--intent_decoder_hidden_dim', '-idhd', type=int, default=64)
parser.add_argument('--attention_hidden_dim', '-ahd', type=int, default=1024)
parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)

parser.add_argument('--info_embedding_dim', '-ined', type=int, default=128)
parser.add_argument('--up_input_dim', '-uid', type=int, default=11)
parser.add_argument('--ca_input_dim', '-cid', type=int, default=18)

args = parser.parse_args()
args.up_keys = ['音视频应用偏好', '出行交通工具偏好', '长途交通工具偏好', '是否有车']
args.ca_keys = ['移动状态', '姿态识别', '地理围栏', '户外围栏']
args.gpu = args.gpu and torch.cuda.is_available()

if not args.use_fitlog:
    fitlog.debug()

timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')
if args.use_pretrained:
    prefix = 'BERTSLU'
else:
    prefix = 'SLU'
    args.model_type = 'LSTM'
if args.use_info:
    prefix += '++'
args.save_dir = os.path.join('save', prefix, '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.use_info,
                                                                                 args.model_type,
                                                                                 args.batch_size,
                                                                                 args.dropout_rate,
                                                                                 args.learning_rate,
                                                                                 args.bert_learning_rate,
                                                                                 args.word_embedding_dim,
                                                                                 args.info_embedding_dim,
                                                                                 timestamp))

os.makedirs(args.save_dir, exist_ok=True)
log_path = os.path.join(args.save_dir, "config.json")
with open(log_path, "w", encoding="utf8") as fw:
    fw.write(json.dumps(args.__dict__, indent=True))

mylogger = get_logger(os.path.join(args.save_dir, 'log.txt'), name='SLU')
mylogger.info(str(vars(args)))

# logger
fitlog.set_log_dir(args.fit_log_dir)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)

# Model Dict
if args.model_type != 'LSTM':
    model_type = {
        'RoBERTa': "PretrainModel/bert/chinese-roberta-wwm-ext",
        'BERT': "PretrainModel/bert/chinese-bert-wwm-ext",
        'XLNet': "PretrainModel/bert/chinese-xlnet-base",
        'ELECTRA': "PretrainModel/bert/chinese-electra-180g-base-discriminator",
    }

    args.model_type_path = model_type[args.model_type]
