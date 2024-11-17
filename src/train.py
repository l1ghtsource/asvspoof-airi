import numpy as np
import torch
import time
import evaluate
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from transformers import (
    ASTConfig,
    ASTForAudioClassification,
    ASTFeatureExtractor,
    WhisperModel,
    WhisperFeatureExtractor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    get_linear_schedule_with_warmup,
    AdamW
)
from audiomentations import (
    Compose,
    AddGaussianSNR,
    GainTransition,
    Gain,
    ClippingDistortion
)
import audiomentations as AA
from torch.utils.data import DataLoader
from datasets import Audio

from data.dataset import get_ast_dataset, SedDataset, WhisperDataset
from modules.losses import PANNsLoss, FocalLoss
from modules.sed_model import AudioSEDModel
from modules.whisper_model import WhisperClassifier
from modules.custom_trainer import FocalTrainer, TimeLimitCallback
from utils.metrics import AverageMeter, MetricMeter
from utils.seed import seed_everything


def train_ast(config):
    dataset = get_ast_dataset()
    dataset = dataset.cast_column('audio', Audio(sampling_rate=config['model']['sampling_rate']))
    num_labels = config['model']['num_labels']
    pretrained_model = config['model']['name']
    feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)
    model_input_name = feature_extractor.model_input_names[0]  # key -> 'input_values'
    SAMPLING_RATE = feature_extractor.sampling_rate

    def preprocess_audio(batch):
        wavs = [audio['array'] for audio in batch['input_values']]
        # inputs are spectrograms as torch.tensors now
        inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors='pt')

        output_batch = {model_input_name: inputs.get(model_input_name), 'labels': list(batch['labels'])}
        return output_batch

    dataset = dataset.rename_column('audio', 'input_values')  # rename audio column
    dataset.set_transform(preprocess_audio, output_all_columns=False)

    feature_extractor.do_normalize = False  # we set normalization to False in order to calculate the mean + std of the dataset
    mean = []
    std = []

    # we use the transformation w/o augmentation on the training dataset to calculate the mean + std
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

    audio_augmentations = Compose([
        # AddGaussianSNR(min_snr_db=10, max_snr_db=20, p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.2),
        GainTransition(
            min_gain_db=-6, max_gain_db=6, min_duration=0.01, max_duration=0.3,
            duration_unit='fraction', p=0.2
        ),
        # ClippingDistortion(
        #     min_percentile_threshold=0, max_percentile_threshold=30, p=0.1
        # ),
    ], p=0.8, shuffle=True)

    def preprocess_audio_with_transforms(batch):
        # we apply augmentations on each waveform
        wavs = [audio_augmentations(audio['array'], sample_rate=SAMPLING_RATE) for audio in batch['input_values']]
        inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors='pt')

        output_batch = {model_input_name: inputs.get(model_input_name), 'labels': list(batch['labels'])}

        return output_batch

    # cast the audio column to the appropriate feature type and rename it
    dataset = dataset.cast_column('audio', Audio(sampling_rate=feature_extractor.sampling_rate))

    # with augmentations on the training set
    dataset['train'].set_transform(preprocess_audio_with_transforms, output_all_columns=False)
    # w/o augmentations on the val set
    dataset['val'].set_transform(preprocess_audio, output_all_columns=False)
    # w/o augmentations on the test set
    dataset['test'].set_transform(preprocess_audio, output_all_columns=False)

    config_ = ASTConfig.from_pretrained(pretrained_model)

    config_.num_labels = num_labels
    config_.label2id = {'real': 0, 'fake': 1}
    config_.id2label = {v: k for k, v in config_.label2id.items()}

    model = ASTForAudioClassification.from_pretrained(pretrained_model, config=config_, ignore_mismatched_sizes=True)
    model.init_weights()

    training_args = TrainingArguments(
        output_dir='./runs/ast_classifier',
        logging_dir='./logs/ast_classifier',
        report_to='wandb',
        learning_rate=config['model']['learning_rate'],
        push_to_hub=False,
        num_train_epochs=config['model']['num_epochs'],
        per_device_train_batch_size=config['model']['batch_size'],
        eval_strategy='epoch',
        save_strategy='steps',
        eval_steps=1,
        save_steps=500,
        metric_for_best_model='accuracy',
        logging_strategy='steps',
        logging_steps=1,
    )

    accuracy = evaluate.load('accuracy')
    recall = evaluate.load('recall')
    precision = evaluate.load('precision')
    f1 = evaluate.load('f1')
    # roc_auc = evaluate.load('roc_auc')

    AVERAGE = 'binary'

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        predictions = np.argmax(logits, axis=1)
        metrics = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
        metrics.update(precision.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
        metrics.update(recall.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
        metrics.update(f1.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
        # metrics.update(roc_auc.compute(predictions=logits[:, 1], references=eval_pred.label_ids))
        return metrics

    # class_weights = [0.7, 1.3] # example

    # trainer = FocalTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset['train'],
    #     eval_dataset=dataset['val'],
    #     compute_metrics=compute_metrics,  # use the metrics function from above
    #     class_weights=class_weights,
    #     callbacks=[TimeLimitCallback(max_time_in_seconds=3600*config['model']['time_limit'])]  # 8 hours
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        compute_metrics=compute_metrics,  # use the metrics function from above
        callbacks=[TimeLimitCallback(max_time_in_seconds=3600*config['model']['time_limit'])]  # 8 hours
    )

    trainer.train()


def train_sed(config):
    def train_epoch_sed(config, model, loader, criterion, optimizer, scheduler, epoch):
        losses = AverageMeter()
        scores = MetricMeter()

        model.train()
        t = tqdm(loader)
        for i, sample in enumerate(t):
            optimizer.zero_grad()
            input = sample['image'].to(config['device'])
            target = sample['target'].to(config['device'])
            # mixup = Mixup(mixup_alpha=0.4)
            # mixup_lambda = mixup.get_lambda(input.size(0) // 2).to(input.device)
            # output = model(input, mixup_lambda)
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if scheduler and config['step_scheduler']:
                scheduler.step()

            bs = input.size(0)
            scores.update(target, torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0]))
            losses.update(loss.item(), bs)

            t.set_description(f"Train E:{epoch} - Loss{losses.avg:0.4f}")
        t.close()
        return scores.avg, losses.avg

    def valid_epoch_sed(config, model, loader, criterion, epoch):
        losses = AverageMeter()
        scores = MetricMeter()
        model.eval()
        with torch.no_grad():
            t = tqdm(loader)
            for i, sample in enumerate(t):
                input = sample['image'].to(config['device'])
                target = sample['target'].to(config['device'])
                output = model(input)
                loss = criterion(output, target)

                bs = input.size(0)
                scores.update(target, torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0]))
                losses.update(loss.item(), bs)
                t.set_description(f"Valid E:{epoch} - Loss:{losses.avg:0.4f}")
        t.close()
        return scores.avg, losses.avg

    seed_everything(config['seed'])

    train_audio_transform = AA.Compose([
        AA.AddGaussianNoise(p=0.5),
        AA.AddGaussianSNR(p=0.5),
        # AA.AddBackgroundNoise("../input/train_audio/", p=1)
        # AA.AddImpulseResponse(p=0.1),
        # AA.AddShortNoises("../input/train_audio/", p=1)
        # AA.FrequencyMask(min_frequency_band=0.0,  max_frequency_band=0.2, p=0.1),
        # AA.TimeMask(min_band_part=0.0, max_band_part=0.2, p=0.1),
        # AA.PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.1),
        # AA.Shift(p=0.1),
        # AA.Normalize(p=0.1),
        # AA.ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=1, p=0.05),
        # AA.PolarityInversion(p=0.05),
        AA.Gain(p=0.2)
    ])

    save_path = os.path.join(config['output_dir'], config['exp_name'])
    os.makedirs(save_path, exist_ok=True)

    train_dataset = SedDataset(
        root_dir=config['data_root'],
        period=config['period'],
        audio_transform=train_audio_transform,
        mode="train"
    )

    valid_dataset = SedDataset(
        root_dir=config['data_root'],
        period=config['period'],
        stride=5,
        audio_transform=None,
        mode="validation"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=config['num_workers']
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=config['num_workers']
    )

    model = AudioSEDModel(**config['model_param'])
    model = model.to(config['device'])

    if config.pretrain_weights:
        print("Loading pretrained weights...")
        model.load_state_dict(torch.load(config['pretrain_weights'], map_location=config['device']), strict=False)
        model = model.to(config.device)

    criterion = PANNsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    num_train_steps = int(len(train_loader) * config['epochs'])
    num_warmup_steps = int(0.1 * config['epochs'] * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    best_metric = -np.inf
    early_stop_count = 0

    for epoch in range(config['start_epoch'], config['epochs']):
        train_avg, train_loss = train_epoch_sed(config, model, train_loader, criterion, optimizer, scheduler, epoch)
        valid_avg, valid_loss = valid_epoch_sed(config, model, valid_loader, criterion, epoch)

        if config['epoch_scheduler']:
            scheduler.step()

        content = f"""
                {time.ctime()} \n
                Epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']:.7}\n
                Train Loss:{train_loss:0.4f} - ROC AUC:{train_avg['roc_auc']:0.4f}, F1:{train_avg['f1']:0.4f}, Accuracy:{train_avg['accuracy']:0.4f}\n
                Valid Loss:{valid_loss:0.4f} - ROC AUC:{valid_avg['roc_auc']:0.4f}, F1:{valid_avg['f1']:0.4f}, Accuracy:{valid_avg['accuracy']:0.4f}\n
        """
        print(content)
        with open(f'{save_path}/log_{config['exp_name']}.txt', 'a') as appender:
            appender.write(content+'\n')

        # save the model if ROC AUC improves
        if valid_avg['roc_auc'] > best_metric:
            print(f"########## >>>>>>>> Model Improved From {best_metric} ----> {valid_avg['roc_auc']}")
            torch.save(model.state_dict(), os.path.join(save_path, 'xdd.bin'))
            best_metric = valid_avg['roc_auc']
            early_stop_count = 0
        else:
            early_stop_count += 1

        if config['early_stop'] == early_stop_count:
            print("\n $$$ ---? Ohoo.... we reached early stopping count :", early_stop_count)
            break


def train_whisper(config):
    def train(model, train_loader, val_loader, optimizer,  criterion, device, num_epochs):
        best_accuracy = 0.0

        for epoch in range(num_epochs):
            model.train()
            for i, batch in enumerate(train_loader):
                input_features, decoder_input_ids, labels = batch
                input_features = input_features.squeeze()
                input_features = input_features.to(device)

                decoder_input_ids = decoder_input_ids.squeeze()
                decoder_input_ids = decoder_input_ids.to(device)

                labels = labels.view(-1)
                labels = labels.to(device)

                optimizer.zero_grad()

                logits = model(input_features, decoder_input_ids)

                loss = criterion(logits, labels)
                loss.backward()

                optimizer.step()

                if (i+1) % 8 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Train Loss: {loss.item() :.4f}')
                    train_loss = 0.0

            val_loss, val_accuracy, val_f1, _, _ = evaluate(model, val_loader, device)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), 'best_model.pt')

            print("===========" * 8)
            print(
                f'Epoch {epoch+1}/{num_epochs},
                Val Loss: {val_loss: .4f},
                Val Accuracy: {val_accuracy: .4f},
                Val F1: {val_f1: .4f},
                Best Accuracy: {best_accuracy: .4f}'
            )
            print("===========" * 8)

    def evaluate(model, data_loader,  device):
        all_labels = []
        all_preds = []
        total_loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                input_features, decoder_input_ids, labels = batch
                input_features = input_features.squeeze()
                input_features = input_features.to(device)

                decoder_input_ids = decoder_input_ids.squeeze()
                decoder_input_ids = decoder_input_ids.to(device)

                labels = labels.view(-1)
                labels = labels.to(device)

                optimizer.zero_grad()

                logits = model(input_features, decoder_input_ids)

                loss = criterion(logits, labels)
                total_loss += loss.item()

                _, preds = torch.max(logits, 1)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        return loss, accuracy, f1, all_labels, all_preds

    dataset = get_ast_dataset()
    dataset = dataset.cast_column('audio', Audio(sampling_rate=config['model']['sampling_rate']))

    model_checkpoint = config['model']['name']
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_checkpoint)
    encoder = WhisperModel.from_pretrained(model_checkpoint)

    device = config['model']['device']

    audio_augmentations = Compose([
        AddGaussianSNR(min_snr_db=10, max_snr_db=20, p=0.5),
        AA.AddGaussianNoise(p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.25),
    ], p=0.8, shuffle=True)

    train_dataset = WhisperDataset(dataset['train'], feature_extractor, encoder, audio_augmentations)
    val_dataset = WhisperDataset(dataset['val'], feature_extractor, encoder)

    train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['model']['batch_size'], shuffle=False)

    model = WhisperClassifier(
        num_labels=config['model']['num_labels'],
        encoder=encoder,
        dropout=config['model']['dropout']
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config['model']['lr'], betas=(0.9, 0.999), eps=1e-08)

    if config['model']['loss'] == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif config['model']['loss'] == 'focal':
        criterion = FocalLoss()
    else:
        raise 'Choose loss from ["ce", "focal"]'

    num_epochs = config['model']['num_epochs']
    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs)
