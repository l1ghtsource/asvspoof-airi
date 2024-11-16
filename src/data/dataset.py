from datasets import Dataset, DatasetDict, Audio, ClassLabel, Features
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
