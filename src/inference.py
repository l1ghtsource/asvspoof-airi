import pandas as pd
import torch
import os
import numpy as np
from tqdm import tqdm

from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    Trainer,
    TrainingArguments
)
from datasets import Audio
from data.dataset import get_ast_dataset


def inference_ast(config):
    pretrained_model = config['model']['name']
    feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)
    model = ASTForAudioClassification.from_pretrained(config['model']['checkpoint'])

    def preprocess_audio(batch):
        wavs = [audio['array'] for audio in batch['input_values']]
        inputs = feature_extractor(wavs, sampling_rate=feature_extractor.sampling_rate, return_tensors='pt')
        return {'input_values': inputs['input_values']}

    feature_extractor.do_normalize = False  # we set normalization to False in order to calculate the mean + std of the dataset
    mean = []
    std = []

    dataset = get_ast_dataset()
    dataset['train'].set_transform(preprocess_audio, output_all_columns=False)

    if config['model']['know_stats']:
        feature_extractor.mean = -4.556313  # np.mean(mean)
        feature_extractor.std = 4.4429035  # p.mean(std)
        feature_extractor.do_normalize = True
    else:
        for i, (audio_input, labels) in tqdm(enumerate(dataset['train']),
                                             total=len(dataset['train']),
                                             desc='Processing Audio'):
            cur_mean = torch.mean(dataset["train"][i][audio_input])
            cur_std = torch.std(dataset["train"][i][audio_input])
            mean.append(cur_mean)
            std.append(cur_std)

        feature_extractor.mean = np.mean(mean)
        feature_extractor.std = np.mean(std)
        feature_extractor.do_normalize = True

    test_dataset = dataset['test']
    test_dataset = test_dataset.cast_column('audio', Audio(sampling_rate=feature_extractor.sampling_rate))

    test_dataset = test_dataset.rename_column('audio', 'input_values')
    test_dataset.set_transform(preprocess_audio, output_all_columns=False)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir='./results',
            per_device_eval_batch_size=128
        )
    )

    test_predictions = trainer.predict(test_dataset)
    logits = test_predictions.predictions
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    class_0 = probabilities[:, 0]
    class_1 = probabilities[:, 1]

    test_dir = config['data']['test_dir']
    test_audio_files = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith('.wav')]
    idxs = [int(test_audio_files[i].split('/')[-1][:-4]) for i in range(len(test_audio_files))]

    df = pd.DataFrame({
        'Id': idxs,
        'Predicted': class_0
    }).sort_values(by=['Id'])

    df['Predicted'] = (df['Predicted'] > 0.5).astype(int)
    df.to_csv('submission_ast.csv', index=False)
