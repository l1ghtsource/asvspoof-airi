import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from timm import create_model

from sed_modules import (
    AttBlock,
    init_layer,
    init_bn,
    interpolate,
    pad_framewise_output
)
from augmentations import do_mixup


encoder_params = {
    'tf_efficientnet_b0_ns_jft_in1k': {
        'features': 1280,
        'init_op': partial(create_model, 'tf_efficientnet_b0.ns_jft_in1k', pretrained=True, drop_path_rate=0.1)
    },
    'resnet50_a1_in1k': {
        'features': 2048,
        'init_op': partial(create_model, 'resnet50.a1_in1k', pretrained=True, drop_path_rate=0.1)
    },
    'resnext50_32x4d_a1h_in1k': {
        'features': 2048,
        'init_op': partial(create_model, 'resnext50_32x4d.a1h_in1k', pretrained=True, drop_path_rate=0.1)
    },
    'regnety_120_sw_in12k_ft_in1k': {
        'features': 2240,
        'init_op': partial(create_model, 'regnety_120.sw_in12k_ft_in1k', pretrained=True, drop_path_rate=0.1)
    },
    'convnext_base_fb_in22k_ft_in1k': {
        'features': 1024,
        'init_op': partial(create_model, 'convnext_base.fb_in22k_ft_in1k', pretrained=True, drop_path_rate=0.1)
    },
}


class AudioSEDModel(nn.Module):
    def __init__(self, encoder, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 30  # downsampled ratio

        # spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True
        )

        # logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True
        )

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2
        )

        # model encoder
        self.encoder = encoder_params[encoder]['init_op']()
        self.fc1 = nn.Linear(encoder_params[encoder]['features'], 1024, bias=True)
        self.att_block = AttBlock(1024, classes_num, activation='sigmoid')
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)

    def forward(self, input, mixup_lambda=None):
        '''Input : (batch_size, data_length'''

        x = self.spectrogram_extractor(input)
        # batch_size x 1 x time_steps x freq_bins
        x = self.logmel_extractor(x)
        # batch_size x 1 x time_steps x mel_bins

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # print(x.shape)

        if self.training:
            x = self.spec_augmenter(x)

        # mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        # output shape (batch size, channels, time, frequency)
        x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3])
        # print(x.shape)
        x = self.encoder.forward_features(x)
        # print(x.shape)
        x = torch.mean(x, dim=3)
        # print(x.shape)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        # print(x.shape)

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        # print(x.shape)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        output_dict = {
            'framewise_output': framewise_output,
            'logit': logit,
            'clipwise_output': clipwise_output
        }

        return output_dict
