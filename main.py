import sys

# SurvTRACE models
from SurvTrace_OneEvent import SurvTrace_OneEvent
from SurvTrace_MultiEvent import SurvTrace_MultiEvent

# Load datasets
from LoadMETABRIC import load_metabric_data
from LoadSUPPORT import load_support_data

# SurvTRACE utils
from evaluate_utils import Evaluator
from train_utils import Trainer
from config import STConfig


def metabric():
    # use METABRIC dataset
    STConfig['data'] = 'metabric'
    df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_metabric_data(STConfig)

    # initialize model
    model = SurvTrace_OneEvent(STConfig)

    # execute training
    trainer = Trainer(model)
    trainer.fit((df_train, df_y_train), (df_val, df_y_val))

    # evaluating
    evaluator = Evaluator(df, df_train.index)
    evaluator.eval(model, (df_test, df_y_test))

    print("done!")


def support():
    # use SUPPORT dataset
    STConfig['data'] = 'support'
    df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_support_data(STConfig)

    # initialize model
    model = SurvTrace_OneEvent(STConfig)

    # execute training
    trainer = Trainer(model)
    trainer.fit((df_train, df_y_train), (df_val, df_y_val))

    # evaluating
    evaluator = Evaluator(df, df_train.index)
    evaluator.eval(model, (df_test, df_y_test))

    print("done!")


dataset = sys.argv[1]

if dataset == 'metabric':
    metabric()
elif dataset == 'support':
    support()
else:
    print('dataset not supported!')
    print('please specify metabric or support')

