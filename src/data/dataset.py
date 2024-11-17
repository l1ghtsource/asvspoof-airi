from datasets import Dataset, DatasetDict, Audio, ClassLabel, Features
import numpy as np
import soundfile as sf
import os


def load_data_from_directory(data_dir, label):
    audio_files = []
    labels = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):
            audio_files.append(os.path.join(data_dir, file_name))
            labels.append(label)
    return audio_files, labels


def get_ast_dataset(
    train_real_dir='competiton/audio_split/train/real',
    train_fake_dir='competiton/audio_split/train/fake',
    val_real_dir='competiton/audio_split/validation/real',
    val_fake_dir='competiton/audio_split/validation/fake',
    test_dir='competiton/audio_split/test'
):
    class_names = ['real', 'fake']
    class_labels = ClassLabel(names=class_names)

    features = Features({
        'audio': Audio(),
        'labels': class_labels
    })

    real_audio_files, real_labels = load_data_from_directory(train_real_dir, label=0)
    fake_audio_files, fake_labels = load_data_from_directory(train_fake_dir, label=1)

    train_audio_files = real_audio_files + fake_audio_files
    train_labels = real_labels + fake_labels

    train_dataset = Dataset.from_dict({
        'audio': train_audio_files,
        'labels': train_labels
    }, features=features)

    val_real_audio_files, val_real_labels = load_data_from_directory(val_real_dir, label=0)
    val_fake_audio_files, val_fake_labels = load_data_from_directory(val_fake_dir, label=1)

    val_audio_files = val_real_audio_files + val_fake_audio_files
    val_labels = val_real_labels + val_fake_labels

    val_dataset = Dataset.from_dict({
        'audio': val_audio_files,
        'labels': val_labels
    }, features=features)

    test_audio_files = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith('.wav')]

    test_dataset = Dataset.from_dict({
        'audio': test_audio_files
    }, features=Features({
        'audio': Audio()
    }))

    dataset = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    })

    return dataset


class SedDataset(Dataset):
    def __init__(self, root_dir, period=10, stride=5, audio_transform=None, mode="train"):
        self.period = period
        self.stride = stride
        self.audio_transform = audio_transform
        self.mode = mode

        self.samples = []

        if mode == "test":
            test_dir = os.path.join(root_dir, 'test')
            for file_name in os.listdir(test_dir):
                if file_name.endswith('.wav'):
                    self.samples.append({
                        'path': os.path.join(test_dir, file_name),
                        'label': None,
                        'id': os.path.splitext(file_name)[0]
                    })
        else:
            mode_dir = os.path.join(root_dir, mode)
            for class_name in ['real', 'fake']:
                class_dir = os.path.join(mode_dir, class_name)
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.wav'):
                        self.samples.append({
                            'path': os.path.join(class_dir, file_name),
                            'label': 0 if class_name == 'real' else 1,
                            'id': os.path.splitext(file_name)[0]
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        y, sr = sf.read(sample['path'])

        # if self.mode == "test":
        #     effective_length = self.period * sr
        #     stride_length = self.stride * sr

        #     n_windows = (len(y) - effective_length) // stride_length + 1

        #     windows = []
        #     for i in range(n_windows):
        #         start = i * stride_length
        #         end = start + effective_length
        #         window = y[start:end]
        #         if len(window) == effective_length:
        #             windows.append(window)

        #     y = np.stack([window.astype(np.float32) for window in windows])
        #     label = np.array([0], dtype=np.float32)

        # else:
        #     if len(y) > self.period * sr:
        #         if self.mode == "train":
        #             start = np.random.randint(0, len(y) - self.period * sr)
        #         else:
        #             start = (len(y) - self.period * sr) // 2
        #         y = y[start:start + self.period * sr]
        #     else:
        #         pad_length = self.period * sr - len(y)
        #         y = np.pad(y, (0, pad_length), mode='constant')

        #     y = y.astype(np.float32)
        #     label = np.array([sample["label"]], dtype=np.float32)

        if self.mode == 'test':
            if len(y) > self.period * sr:
                start = (len(y) - self.period * sr) // 2
                y = y[start:start + self.period * sr]
            else:
                pad_length = self.period * sr - len(y)
                y = np.pad(y, (0, pad_length), mode='constant')

            y = y.astype(np.float32)
            label = np.array([0], dtype=np.float32)
        else:
            if len(y) > self.period * sr:
                if self.mode == 'train':
                    start = np.random.randint(0, len(y) - self.period * sr)
                else:
                    start = (len(y) - self.period * sr) // 2
                y = y[start:start + self.period * sr]
            else:
                pad_length = self.period * sr - len(y)
                y = np.pad(y, (0, pad_length), mode='constant')

            y = y.astype(np.float32)
            label = np.array([sample['label']], dtype=np.float32)

        if self.audio_transform:
            y = self.audio_transform(samples=y, sample_rate=sr)

        return {
            'image': y,
            'target': label,
            'id': sample['id']
        }
