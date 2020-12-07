import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel, ClassificationArgs


def get_args():
    model_args = ClassificationArgs()
    model_args.labels_list = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    model_args.cache_dir = 'cache/'
    model_args.output_dir = 'outputs'
    model_args.overwrite_output_dir = True
    model_args.n_gpu = 2
    model_args.learning_rate = 1e-5
    model_args.num_train_epochs = 10
    model_args.train_batch_size = 128
    model_args.eval_batch_size = 128

    return model_args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    dataset_dir = 'dm2020-hw2-nthu'
    dataset_path = Path(dataset_dir)
    train_df = pickle.load(open(dataset_path/'train.pkl', 'rb'))
    test_df = pickle.load(open(dataset_path/'test.pkl', 'rb'))

    x_train, x_val, y_train, y_val = train_test_split(
        train_df['text'].values, train_df['emotion'].values, test_size=0.2, shuffle=True, random_state=55688)

    train_data = pd.DataFrame({'id': x_train, 'label': y_train})
    val_data = pd.DataFrame({'id': x_val, 'label': y_val})

    model = ClassificationModel('roberta', 'roberta-base', args=get_args(), num_labels=8)

    # Train the model
    model.train_model(train_data)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(val_data)

    predictions, _ = model.predict(test_df['text'].values)

    submission_df = pd.DataFrame({'id': test_df['tweet_id'].values, 'emotion': predictions})
    submission_df.to_csv('submission.csv', index=False)
