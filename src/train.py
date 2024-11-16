import numpy as np
import torch
import time
import evaluate
from tqdm import tqdm

from transformers import (
    ASTConfig,
    ASTForAudioClassification,
    ASTFeatureExtractor,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from audiomentations import (
    Compose,
    AddGaussianSNR,
    GainTransition,
    Gain,
    ClippingDistortion
)
from datasets import Audio
from data.dataset import get_ast_dataset


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
        AddGaussianSNR(min_snr_db=10, max_snr_db=20, p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.2),
        GainTransition(
            min_gain_db=-6, max_gain_db=6, min_duration=0.01, max_duration=0.3,
            duration_unit='fraction', p=0.2
        ),
        ClippingDistortion(
            min_percentile_threshold=0, max_percentile_threshold=30, p=0.1
        ),
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
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_strategy='steps',
        logging_steps=1,
    )

    accuracy = evaluate.load('accuracy')
    recall = evaluate.load('recall')
    precision = evaluate.load('precision')
    f1 = evaluate.load('f1')

    AVERAGE = 'macro' if config_.num_labels > 2 else 'binary'

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        predictions = np.argmax(logits, axis=1)
        metrics = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
        metrics.update(precision.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
        metrics.update(recall.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
        metrics.update(f1.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
        return metrics

    class TimeLimitCallback(TrainerCallback):
        def __init__(self, max_time_in_seconds):
            self.max_time_in_seconds = max_time_in_seconds
            self.start_time = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.start_time = time.time()  # start the timer when training begins

        def on_step_end(self, args, state, control, **kwargs):
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.max_time_in_seconds:
                print(f"Stopping training after {self.max_time_in_seconds / 3600} hours.")
                control.should_early_stop = True  # stop the training
                control.should_save = True  # optionally save the model at the end

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        compute_metrics=compute_metrics,  # use the metrics function from above
        callbacks=[TimeLimitCallback(max_time_in_seconds=3600*config['model']['time_limit'])]  # 8 hours
    )

    trainer.train()
