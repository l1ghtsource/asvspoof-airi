import pandas as pd
import torch
import os
import numpy as np
from tqdm import tqdm

from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    WhisperFeatureExtractor,
    WhisperModel,
    Trainer,
    TrainingArguments
)
from torch.utils.data import DataLoader
from datasets import Audio

from data.dataset import get_ast_dataset, SedDataset, WhisperTestDataset
from src.modules.sed_model import AudioSEDModel
from src.modules.whisper_model import WhisperClassifier
from utils.seed import seed_everything


def inference_ast(config, checkpoint):
    feature_extractor = ASTFeatureExtractor.from_pretrained(checkpoint)
    model = ASTForAudioClassification.from_pretrained(checkpoint)

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


def inference_sed(config, checkpoint):
    def test_epoch(config, model, loader):
        model.eval()
        pred_list = []
        id_list = []
        with torch.no_grad():
            t = tqdm(loader)
            for i, sample in enumerate(t):
                input = sample['image'].to(config['device'])
                id = sample['id']
                output = model(input)
                output = torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0])
                output = [x[0] for x in output.cpu().numpy().tolist()]
                pred_list.extend(output)
                id_list.extend(id)

        return pred_list, id_list

    seed_everything(config['seed'])

    test_dataset = SedDataset(
        root_dir=config['data_root'],
        period=config['period'],
        stride=5,
        audio_transform=None,
        mode="test"
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=config['num_workers']
    )

    model = AudioSEDModel(**config['model_param'])
    model = model.to(config['device'])

    if config['pretrain_weights']:
        print("Loading pretrained weights...")
        model.load_state_dict(torch.load(config['pretrain_weights'], map_location=config['device']), strict=False)
        model = model.to(config.device)

    model.load_state_dict(torch.load(os.path.join(checkpoint), map_location=config['device']))

    test_pred, ids = test_epoch(config, model, test_loader)

    test_pred_df = pd.DataFrame({
        'Id': ids,
        'Predicted': test_pred
    })
    test_pred_df['Id'] = test_pred_df['Id'].astype(int)
    test_pred_df = test_pred_df.sort_values(by='Id')

    thold = config['thold']
    test_pred_df['Predicted'] = (test_pred_df['Predicted'] < thold).astype(int)
    test_pred_df.to_csv('submission.csv', index=False)


def inference_whisper(config, checkpoint):
    def inference(model, data_loader, device):
        all_preds = []

        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                input_features, decoder_input_ids = batch
                input_features = input_features.squeeze()
                input_features = input_features.to(device)

                decoder_input_ids = decoder_input_ids.squeeze()
                decoder_input_ids = decoder_input_ids.to(device)

                logits = model(input_features, decoder_input_ids)
                _, preds = torch.max(logits, 1)
                all_preds.append(preds.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        return all_preds

    dataset = get_ast_dataset()
    dataset = dataset.cast_column('audio', Audio(sampling_rate=config['model']['sampling_rate']))

    model_checkpoint = config['model']['name']
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_checkpoint)
    encoder = WhisperModel.from_pretrained(model_checkpoint)

    device = config['model']['device']

    test_dataset = WhisperTestDataset(dataset['test'], feature_extractor)
    test_loader = DataLoader(test_dataset, batch_size=config['model']['batch_size'], shuffle=False)

    model = WhisperClassifier(
        num_labels=config['model']['num_labels'],
        encoder=encoder,
        dropout=config['model']['dropout']
    ).to(device)

    model.load_state_dict(
        torch.load(checkpoint, map_location=device)
    )

    preds = inference(model, test_loader, device)
