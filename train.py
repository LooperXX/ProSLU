import random
import numpy as np

from models.model import ModelManager
from utils.config import *
from utils.loader import DatasetManager
from utils.process import Processor

if __name__ == "__main__":
    if args.fix_seed:
        # Fix the random seed of package random.
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

        # Fix the random seed of Pytorch when using CPU.
        torch.manual_seed(args.random_seed)
        torch.random.manual_seed(args.random_seed)

        # Fix the random seed of Pytorch when using GPU.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
            torch.cuda.manual_seed(args.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Instantiate a dataset object.
    dataset = DatasetManager(args)
    dataset.quick_build()

    # Instantiate a network model object.
    model = ModelManager(
        args, len(dataset.word_alphabet),
        len(dataset.slot_alphabet),
        len(dataset.intent_alphabet))

    # To train and evaluate the models.
    process = Processor(dataset, model, args)
    process.train()

    mylogger.info('\nAccepted performance: ' + str(process.validate(
        os.path.join(args.save_dir, "model/model.pkl"),
        os.path.join(args.save_dir, "model/dataset.pkl")
    )) + " at test dataset;\n")
