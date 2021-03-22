from typing import Tuple

import cv2 as cv
import numpy as np
import torch
import torch.nn.init as init
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models.squeezenet import Fire


def get_output_shape(layer, shape):
    layer_training = layer.training
    if layer_training:
        layer.eval()
    out = layer(torch.zeros(1, *shape))
    before_flattening = tuple(out.size())[1:]
    after_flattening = int(np.prod(out.size()))
    if layer_training:
        layer.train()
    return before_flattening, after_flattening


class CNNPreProcessor(object):

    def __init__(self, bgr_mean=(0.5, 0.5, 0.5), bgr_std=(0.25, 0.25, 0.25), dtype=torch.float32, resize_factor=1.0,
                 target_shape=None, to_grayscale=False):
        super(CNNPreProcessor, self).__init__()
        self.bgr_mean = np.array(bgr_mean)
        self.bgr_std = np.array(bgr_std)
        self.resize_factor = resize_factor
        self.target_shape = tuple(target_shape) if target_shape is not None else None
        self.to_grayscale = to_grayscale
        self.dtype = dtype

    def prepare_im(self, im):
        if len(im.shape) == 3 and self.to_grayscale:
            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        if self.target_shape is not None:
            im = cv.resize(im, self.target_shape)
        elif self.resize_factor != 1.0:
            im = cv.resize(im, None, fx=self.resize_factor, fy=self.resize_factor)
        return im

    def preprocess(self, im):
        im_t = im.copy()
        im_t = self.prepare_im(im_t)
        im_t = im_t / 255
        im_t = (im_t - self.bgr_mean) / self.bgr_std
        if len(im_t.shape) == 2:
            im_t = im_t[:, :, np.newaxis]
        return torch.from_numpy(im_t.transpose(2, 0, 1)[np.newaxis, :, :]).type(self.dtype)

    def preprocess_im_sequence(self, seq):
        return torch.cat([self.preprocess(el).unsqueeze(1) for el in seq])

    def reverse_preprocess(self, tensor, output_shape=None):
        im = tensor.detach().cpu().numpy().squeeze()
        if len(im.shape) == 3:
            im = im.transpose((1, 2, 0))
        im = ((im * self.bgr_std) + self.bgr_mean) * 255
        im = np.clip(im, 0, 255).astype(np.uint8)
        if output_shape is not None:
            im = cv.resize(im, output_shape)
        elif self.resize_factor != 1.0:
            im = cv.resize(im, None, fx=(1 / self.resize_factor), fy=(1 / self.resize_factor))
        return im


def build_basic_encoder(num_in_channels=3, hidden_channels=(32, 64, 128, 256, 512), kernel_size=3, stride=2,
                        padding=1):
    layers = []

    in_c = num_in_channels
    for h_dim in hidden_channels:
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_c, out_channels=h_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU())
        )
        in_c = h_dim

    return nn.Sequential(*layers)


def build_basic_decoder(num_out_channels=3, decoder_filters=(512, 256, 128, 64, 32), kernel_size=3, stride=2,
                        padding=1, output_padding=1):
    layers = []

    for i in range(len(decoder_filters) - 1):
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(decoder_filters[i],
                                   decoder_filters[i + 1],
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   output_padding=output_padding),
                nn.BatchNorm2d(decoder_filters[i + 1]),
                nn.LeakyReLU())
        )

    layers.append(nn.ConvTranspose2d(decoder_filters[-1], num_out_channels, kernel_size=kernel_size, stride=stride,
                                     padding=padding, output_padding=output_padding))
    return nn.Sequential(*layers)


class SqueezeEncoder(nn.Module):

    def __init__(self, num_in_channels=3, version="1_1"):
        super().__init__()
        self.version = version
        if version == "1_0":
            self.conv1 = nn.Conv2d(num_in_channels, 96, kernel_size=7, stride=2)
            self.relu1 = nn.ReLU(inplace=True)
            self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            self.fire1 = Fire(96, 16, 64, 64)
            self.fire2 = Fire(128, 16, 64, 64)
            self.fire3 = Fire(128, 32, 128, 128)
            self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            self.fire4 = Fire(256, 32, 128, 128)
            self.fire5 = Fire(256, 48, 192, 192)
            self.fire6 = Fire(384, 48, 192, 192)
            self.fire7 = Fire(384, 64, 256, 256)
            self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            self.fire8 = Fire(512, 64, 256, 256)
        elif version == "1_1":
            self.conv1 = nn.Conv2d(num_in_channels, 64, kernel_size=3, stride=2)
            self.relu1 = nn.ReLU(inplace=True)
            self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            self.fire1 = Fire(64, 16, 64, 64)
            self.fire2 = Fire(128, 16, 64, 64)
            self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            self.fire3 = Fire(128, 32, 128, 128)
            self.fire4 = Fire(256, 32, 128, 128)
            self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            self.fire5 = Fire(256, 48, 192, 192)
            self.fire6 = Fire(384, 48, 192, 192)
            self.fire7 = Fire(384, 64, 256, 256)
            self.fire8 = Fire(512, 64, 256, 256)
        else:
            raise ValueError(f"Unsupported version {version}: 1_0 or 1_1 expected")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        if self.version == "1_0":
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.max_pool1(x)
            x = self.fire1(x)
            x = self.fire2(x)
            x = self.fire3(x)
            x = self.max_pool2(x)
            x = self.fire4(x)
            x = self.fire5(x)
            x = self.fire6(x)
            x = self.fire7(x)
            x = self.max_pool3(x)
            x = self.fire8(x)
            return x
        elif self.version == "1_1":
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.max_pool1(x)
            x = self.fire1(x)
            x = self.fire2(x)
            x = self.max_pool2(x)
            x = self.fire3(x)
            x = self.fire4(x)
            x = self.max_pool3(x)
            x = self.fire5(x)
            x = self.fire6(x)
            x = self.fire7(x)
            x = self.fire8(x)
            return x
        raise ValueError(f"Unsupported version {self.version}: 1_0 or 1_1 expected")


class ResidualEncodeBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_planes)
        ) if stride != 1 or in_planes != out_planes else lambda x: x
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualDecodeBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                                        output_padding=stride - 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, output_padding=stride - 1,
                               bias=False),
            nn.BatchNorm2d(out_planes)
        ) if stride != 1 or in_planes != out_planes else lambda x: x
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.up_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualEncoder(nn.Module):

    def __init__(self, num_input_channels=3, blocks_per_layer=None):
        super().__init__()
        if blocks_per_layer is None:
            blocks_per_layer = (2, 2, 2, 2)

        self.curr_planes = 64
        self.num_input_channels = num_input_channels

        self.conv1 = nn.Conv2d(self.num_input_channels, self.curr_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.curr_planes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks_per_layer[0])
        self.layer2 = self._make_layer(128, blocks_per_layer[1], stride=2)
        self.layer3 = self._make_layer(256, blocks_per_layer[2], stride=2)
        self.layer4 = self._make_layer(512, blocks_per_layer[3], stride=2)

    def _make_layer(self, out_planes, num_blocks, stride=1):
        layers = [ResidualEncodeBlock(self.curr_planes, out_planes, stride)]
        self.curr_planes = out_planes
        for _ in range(1, num_blocks):
            layers.append(ResidualEncodeBlock(self.curr_planes, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResidualDecoder(nn.Module):

    def __init__(self, num_output_channels=3, blocks_per_layer=None):
        super().__init__()
        if blocks_per_layer is None:
            blocks_per_layer = (2, 2, 2, 2)

        self.curr_planes = 512
        self.num_output_channels = num_output_channels

        self.up_sample = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.layer1 = self._make_layer(512, blocks_per_layer[0], stride=2)
        self.layer2 = self._make_layer(256, blocks_per_layer[1], stride=2)
        self.layer3 = self._make_layer(128, blocks_per_layer[2], stride=2)
        self.layer4 = self._make_layer(64, blocks_per_layer[3])
        self.final_conv = nn.ConvTranspose2d(self.curr_planes, self.num_output_channels, kernel_size=7, stride=2,
                                             padding=3, bias=False, output_padding=1)

    def _make_layer(self, out_planes, num_blocks, stride=1):
        layers = [ResidualDecodeBlock(self.curr_planes, out_planes, stride)]
        self.curr_planes = out_planes
        for _ in range(1, num_blocks):
            layers.append(ResidualDecodeBlock(self.curr_planes, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.up_sample(x)
        x = self.final_conv(x)

        return x


class AutoEncoder(nn.Module):

    def __init__(self, input_shape, encoder, decoder, latent_dim, decoder_input_size=None) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = encoder
        encoder_out_size, encoder_out_size_flat = get_output_shape(self.encoder, self.input_shape)
        if decoder_input_size is None:
            self.decoder_input_size = encoder_out_size
            self.decoder_input_size_flat = encoder_out_size_flat
        else:
            self.decoder_input_size = decoder_input_size
            self.decoder_input_size_flat = int(np.prod(decoder_input_size))

        self.latent = nn.Linear(encoder_out_size_flat, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.decoder_input_size_flat)
        self.decoder = decoder

    @classmethod
    def get_basic_ae(cls, input_shape=(3, 64, 64), encoder_filters=(32, 64, 128, 256, 512), latent_dim=128):
        num_in_channels = input_shape[0]
        basic_encoder = build_basic_encoder(num_in_channels, encoder_filters)
        basic_decoder = build_basic_decoder(num_in_channels, list(reversed(encoder_filters)))
        basic_ae = AutoEncoder(input_shape, basic_encoder, basic_decoder, latent_dim)
        return basic_ae

    @classmethod
    def get_residual_ae(cls, input_shape=(3, 64, 64), blocks_per_layer=(2, 2, 2, 2), latent_dim=128):
        num_in_channels = input_shape[0]
        encoder = ResidualEncoder(num_in_channels, blocks_per_layer)
        decoder = ResidualDecoder(num_in_channels, list(reversed(blocks_per_layer)))
        ae = AutoEncoder(input_shape, encoder, decoder, latent_dim)
        return ae

    @classmethod
    def get_squeeze_ae(cls, input_shape=(3, 64, 64), decoder_filters=(512, 256, 128, 64, 32),
                       latent_dim=128, decoder_input_size=(512, 2, 2), squeeze_version="1_1"):
        num_in_channels = input_shape[0]
        squeeze_encoder = SqueezeEncoder(num_in_channels, squeeze_version)
        custom_decoder = build_basic_decoder(num_in_channels, list(decoder_filters))
        ae = cls(input_shape, squeeze_encoder, custom_decoder,
                 latent_dim, decoder_input_size=decoder_input_size)
        return ae

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.encoder(x)
        x_flat = x.view((x.size()[0], -1))
        latent = self.latent(x_flat)
        x = self.decoder_input(latent)
        x = x.view((-1,) + self.decoder_input_size)
        return self.decoder(x), latent

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]


class VariationalAutoEncoder(nn.Module):

    def __init__(self, input_shape, encoder, decoder, latent_dim, decoder_input_size=None):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = encoder
        encoder_out_size, encoder_out_size_flat = get_output_shape(self.encoder, self.input_shape)
        if decoder_input_size is None:
            self.decoder_input_size = encoder_out_size
            self.decoder_input_size_flat = encoder_out_size_flat
        else:
            self.decoder_input_size = decoder_input_size
            self.decoder_input_size_flat = int(np.prod(decoder_input_size))

        self.mu = nn.Linear(encoder_out_size_flat, latent_dim)
        self.log_var = nn.Linear(encoder_out_size_flat, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.decoder_input_size_flat)
        self.decoder = decoder

    @classmethod
    def get_basic_vae(cls, input_shape=(3, 64, 64), encoder_filters=(32, 64, 128, 256, 512), latent_dim=128):
        num_in_channels = input_shape[0]
        basic_encoder = build_basic_encoder(num_in_channels, encoder_filters)
        basic_decoder = build_basic_decoder(num_in_channels, list(reversed(encoder_filters)))
        basic_ae = cls(input_shape, basic_encoder, basic_decoder, latent_dim)
        return basic_ae

    @classmethod
    def get_residual_vae(cls, input_shape=(3, 64, 64), blocks_per_layer=(2, 2, 2, 2), latent_dim=128):
        num_in_channels = input_shape[0]
        encoder = ResidualEncoder(num_in_channels, blocks_per_layer)
        decoder = ResidualDecoder(num_in_channels, list(reversed(blocks_per_layer)))
        ae = cls(input_shape, encoder, decoder, latent_dim)
        return ae

    @classmethod
    def get_squeeze_vae(cls, input_shape=(3, 64, 64), decoder_filters=(512, 256, 128, 64, 32),
                        latent_dim=128, decoder_input_size=(512, 2, 2), squeeze_version="1_1"):
        num_in_channels = input_shape[0]
        squeeze_encoder = SqueezeEncoder(num_in_channels, squeeze_version)
        decoder = build_basic_decoder(num_in_channels, list(decoder_filters))
        vae = cls(input_shape, squeeze_encoder, decoder, latent_dim, decoder_input_size=decoder_input_size)
        return vae

    def sample(self, mu, log_var):
        std_dev = torch.exp(0.5 * log_var)
        return torch.randn_like(log_var) * std_dev + mu

    def generate(self, device, num_samples=1):
        random_encodings = torch.randn((num_samples, self.latent_dim)).to(device)
        x = self.decoder_input(random_encodings)
        x = x.view((-1,) + self.decoder_input_size)
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.encoder(x)
        x_flat = x.view((x.size()[0], -1))

        mu = self.mu(x_flat)
        log_var = self.log_var(x_flat)

        sampled = self.sample(mu, log_var)

        x = self.decoder_input(sampled)
        x = x.view((-1,) + self.decoder_input_size)
        return self.decoder(x), mu, log_var

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]


class SequencePredictor(nn.Module):

    def __init__(self, input_shape, encoder, decoder, rnn_dim, rnn_layers=1, bidirectional_rnn=False, action_dim=0,
                 decoder_input_size=None):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = rnn_dim
        self.action_dim = action_dim
        self.encoder = encoder
        self.bidirectional_rnn = bidirectional_rnn
        encoder_out_size, encoder_out_size_flat = get_output_shape(self.encoder, self.input_shape)
        if decoder_input_size is None:
            self.decoder_input_size = encoder_out_size
            self.decoder_input_size_flat = encoder_out_size_flat
        else:
            self.decoder_input_size = decoder_input_size
            self.decoder_input_size_flat = int(np.prod(decoder_input_size))

        self.num_rnn_directions = 2 if self.bidirectional_rnn else 1
        self.num_rnn_layers = rnn_layers
        self.latent = nn.Linear(encoder_out_size_flat, self.latent_dim)
        self.rnn = nn.LSTM(input_size=self.latent_dim + self.action_dim, hidden_size=self.latent_dim + self.action_dim,
                           num_layers=rnn_layers, bidirectional=bidirectional_rnn)

        self.decoder_input = nn.Linear(self.num_rnn_directions * (self.latent_dim + self.action_dim),
                                       self.decoder_input_size_flat)
        self.decoder = decoder

    @classmethod
    def get_basic_seq_predictor(cls, input_shape=(3, 64, 64), encoder_filters=(32, 64, 128, 256, 512), latent_dim=128,
                                rnn_layers=1, bidirectional_rnn=False, action_dim=0):
        num_in_channels = input_shape[0]
        encoder = build_basic_encoder(num_in_channels, encoder_filters)
        decoder = build_basic_decoder(num_in_channels, list(reversed(encoder_filters)))
        return cls(input_shape, encoder, decoder, latent_dim, rnn_layers=rnn_layers,
                   bidirectional_rnn=bidirectional_rnn, action_dim=action_dim)

    @classmethod
    def get_residual_seq_predictor(cls, input_shape=(3, 64, 64), blocks_per_layer=(2, 2, 2, 2), latent_dim=128,
                                   rnn_layers=1, bidirectional_rnn=False, action_dim=0):
        num_in_channels = input_shape[0]
        encoder = ResidualEncoder(num_in_channels, blocks_per_layer)
        decoder = ResidualDecoder(num_in_channels, list(reversed(blocks_per_layer)))
        return cls(input_shape, encoder, decoder, latent_dim, rnn_layers=rnn_layers,
                   bidirectional_rnn=bidirectional_rnn, action_dim=action_dim)

    @classmethod
    def get_squeeze_seq_predictor(cls, input_shape=(3, 64, 64), decoder_filters=(512, 256, 128, 64, 32), latent_dim=128,
                                  decoder_input_size=(512, 2, 2), squeeze_version="1_1", rnn_layers=1,
                                  bidirectional_rnn=False, action_dim=0):
        num_in_channels = input_shape[0]
        encoder = SqueezeEncoder(num_in_channels, squeeze_version)
        decoder = build_basic_decoder(num_in_channels, list(decoder_filters))
        return cls(input_shape, encoder, decoder, latent_dim, rnn_layers=rnn_layers,
                   bidirectional_rnn=bidirectional_rnn, action_dim=action_dim, decoder_input_size=decoder_input_size)

    def forward(self, x, actions=None):
        seq_len, batch_size = x.size()[:2]
        encoder_outs = [self.encoder(x[idx, :, :, :, :]) for idx in range(seq_len)]
        latent_outs = [self.latent(el.view(batch_size, -1)) for el in encoder_outs]

        rnn_in = torch.cat([el.unsqueeze(0) for el in latent_outs])
        if actions is not None and len(actions) > 0:
            rnn_in = torch.cat([rnn_in, actions], dim=2)
        _, (h_n, _) = self.rnn(rnn_in)

        decoder_input_in = h_n.transpose(0, 1).contiguous()
        decoder_input_in = decoder_input_in.view(batch_size, self.num_rnn_layers, self.num_rnn_directions, -1)
        decoder_input_in = decoder_input_in[:, -1, :, :].view(batch_size, -1)
        decoder_out = self.decoder_input(decoder_input_in)
        decoder_out = decoder_out.view((-1,) + self.decoder_input_size)
        return self.decoder(decoder_out)


class DeepVariationalBayesFilter(nn.Module):

    def __init__(self, input_shape, encoder, decoder, latent_dim, action_dim, decoder_input_size=None,
                 num_matrices=4, weight_network_hidden_size=64, rnn_layers=1, bidirectional_rnn=False,
                 initial_hidden_size=None, action_encoder=None):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.use_actions = action_dim > 0
        self.encoder = encoder
        self.action_encoder = lambda x: x if action_encoder is None else action_encoder
        encoder_out_size, encoder_out_size_flat = get_output_shape(self.encoder, self.input_shape)
        if decoder_input_size is None:
            self.decoder_input_size = encoder_out_size
            self.decoder_input_size_flat = encoder_out_size_flat
        else:
            self.decoder_input_size = decoder_input_size
            self.decoder_input_size_flat = int(np.prod(decoder_input_size))

        self.w_mu = nn.Linear(encoder_out_size_flat, latent_dim)
        self.w_log_var = nn.Linear(encoder_out_size_flat, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.decoder_input_size_flat)
        self.decoder = decoder
        self.num_matrices = num_matrices
        self.a_i_list = []
        self.b_i_list = []
        self.c_i_list = []
        self.initialize_matrices()

        self.weight_network_hidden_size = weight_network_hidden_size
        self.matrix_weight_network = nn.Sequential(
            nn.Linear(self.latent_dim + self.action_dim, self.weight_network_hidden_size),
            nn.ELU(),
            nn.Linear(self.weight_network_hidden_size, self.num_matrices),
            nn.Softmax(dim=-1)
        )

        self.initial_hidden_size = initial_hidden_size if initial_hidden_size is not None else self.latent_dim
        self.initial_z_model = nn.Sequential(
            nn.Linear(self.latent_dim, self.initial_hidden_size),
            nn.ELU(),
            nn.Linear(self.initial_hidden_size, self.latent_dim)
        )

        self.num_rnn_layers = rnn_layers
        self.num_rnn_directions = 2 if bidirectional_rnn else 1
        self.to_initial_w_in = nn.Linear(encoder_out_size_flat, self.latent_dim)
        self.initial_w_model = nn.LSTM(input_size=self.latent_dim, hidden_size=self.latent_dim, num_layers=rnn_layers,
                                       bidirectional=bidirectional_rnn)
        self.w_initial_mu = nn.Linear(self.num_rnn_directions * latent_dim, latent_dim)
        self.w_initial_log_var = nn.Linear(self.num_rnn_directions * latent_dim, latent_dim)

    @classmethod
    def get_basic_dvbf_ll(cls, input_shape=(3, 64, 64), encoder_filters=(32, 64, 128, 256, 512), latent_dim=128,
                          rnn_layers=1, bidirectional_rnn=False, action_dim=0, action_encoder=None,
                          weight_network_hidden_size=64, initial_hidden_size=None):
        num_in_channels = input_shape[0]
        encoder = build_basic_encoder(num_in_channels, encoder_filters)
        decoder = build_basic_decoder(num_in_channels, list(reversed(encoder_filters)))
        return cls(input_shape, encoder, decoder, latent_dim, action_dim, rnn_layers=rnn_layers,
                   bidirectional_rnn=bidirectional_rnn, action_encoder=action_encoder,
                   weight_network_hidden_size=weight_network_hidden_size, initial_hidden_size=initial_hidden_size)

    @classmethod
    def get_residual_dvbf_ll(cls, input_shape=(3, 64, 64), blocks_per_layer=(2, 2, 2, 2), latent_dim=128,
                             rnn_layers=1, bidirectional_rnn=False, action_dim=0, action_encoder=None,
                             weight_network_hidden_size=64, initial_hidden_size=None):
        num_in_channels = input_shape[0]
        encoder = ResidualEncoder(num_in_channels, blocks_per_layer)
        decoder = ResidualDecoder(num_in_channels, list(reversed(blocks_per_layer)))
        return cls(input_shape, encoder, decoder, latent_dim, action_dim, rnn_layers=rnn_layers,
                   bidirectional_rnn=bidirectional_rnn, action_encoder=action_encoder,
                   weight_network_hidden_size=weight_network_hidden_size, initial_hidden_size=initial_hidden_size)

    @classmethod
    def get_squeeze_dvbf_ll(cls, input_shape=(3, 64, 64), decoder_filters=(512, 256, 128, 64, 32), latent_dim=128,
                            decoder_input_size=(512, 2, 2), squeeze_version="1_1", rnn_layers=1,
                            bidirectional_rnn=False, action_dim=0, action_encoder=None,
                            weight_network_hidden_size=64, initial_hidden_size=None):
        num_in_channels = input_shape[0]
        encoder = SqueezeEncoder(num_in_channels, squeeze_version)
        decoder = build_basic_decoder(num_in_channels, list(decoder_filters))
        return cls(input_shape, encoder, decoder, latent_dim, action_dim, rnn_layers=rnn_layers,
                   bidirectional_rnn=bidirectional_rnn, action_encoder=action_encoder,
                   decoder_input_size=decoder_input_size, weight_network_hidden_size=weight_network_hidden_size,
                   initial_hidden_size=initial_hidden_size)

    def get_w(self, encoder_out_flat):
        mu = self.w_mu(encoder_out_flat)
        log_var = self.w_log_var(encoder_out_flat)
        return self.sample(mu, log_var), mu, log_var

    def get_next_z(self, z, w, u=None):
        model_in = z if u is None else torch.cat([z, u], dim=-1)
        alpha = self.matrix_weight_network(model_in)

        next_z_list = []

        for b_idx in range(alpha.shape[0]):
            curr_z = z[b_idx]

            a_matrix = torch.sum(torch.stack([alpha[b_idx, j] * self.a_i_list[j] for j in range(self.num_matrices)]),
                                 dim=0)
            if self.use_actions:
                b_matrix = torch.sum(
                    torch.stack([alpha[b_idx, j] * self.b_i_list[j] for j in range(self.num_matrices)]), dim=0)
            else:
                b_matrix = None
            c_matrix = torch.sum(torch.stack([alpha[b_idx, j] * self.c_i_list[j] for j in range(self.num_matrices)]),
                                 dim=0)
            if isinstance(b_matrix, torch.Tensor):
                curr_u = u[b_idx]
                next_z = a_matrix.matmul(curr_z) + b_matrix.matmul(curr_u) + c_matrix.matmul(w[b_idx])
            else:
                next_z = a_matrix.matmul(z) + c_matrix.matmul(w[b_idx])

            next_z_list.append(next_z)
        return torch.stack(next_z_list)

    def initialize_matrices(self):
        for idx in range(self.num_matrices):
            # TODO test different initializations as well
            a_m = nn.Parameter(torch.Tensor(self.latent_dim, self.latent_dim))
            init.kaiming_uniform_(a_m)
            self.register_parameter(f"A_{idx}", a_m)
            self.a_i_list.append(a_m)

            if self.use_actions:
                b_m = nn.Parameter(torch.Tensor(self.latent_dim, self.action_dim))
                init.kaiming_uniform_(b_m)
                self.register_parameter(f"B_{idx}", b_m)
                self.b_i_list.append(b_m)

            c_m = nn.Parameter(torch.Tensor(self.latent_dim, self.latent_dim))
            self.register_parameter(f"C_{idx}", c_m)
            init.kaiming_uniform_(c_m)
            self.c_i_list.append(c_m)

    def get_initial_w(self, encoder_out_flats):
        batch_size = encoder_out_flats[0].shape[0]
        rnn_ins = [self.to_initial_w_in(el) for el in encoder_out_flats]
        rnn_in = torch.stack(rnn_ins)

        _, (h_n, _) = self.initial_w_model(rnn_in)
        decoder_input_in = h_n.transpose(0, 1).contiguous()
        decoder_input_in = decoder_input_in.view(batch_size, self.num_rnn_layers, self.num_rnn_directions, -1)
        decoder_input_in = decoder_input_in[:, -1, :, :].view(batch_size, -1)  # Use only last layer

        w_mu = self.w_initial_mu(decoder_input_in)
        w_log_var = self.w_initial_log_var(decoder_input_in)

        w = self.sample(w_mu, w_log_var)
        return w, w_mu, w_log_var

    def sample(self, mu, log_var):
        std_dev = torch.exp(0.5 * log_var)
        return torch.randn_like(log_var) * std_dev + mu

    def get_initial_z(self, w):
        return self.initial_z_model(w)

    def forward(self, x, actions=None):
        seq_len, batch_size = x.size()[:2]
        encoder_outs = [self.encoder(x[idx, :, :, :, :]) for idx in range(seq_len)]
        encoder_out_flats = [el.view(batch_size, -1) for el in encoder_outs]

        initial_w, initial_w_mu, initial_w_log_var = self.get_initial_w(encoder_out_flats)
        initial_z = self.get_initial_z(initial_w)

        w_mus = [initial_w_mu]
        w_log_vars = [initial_w_log_var]
        z_s = [initial_z]
        w_s = [initial_w]
        u_s = None if actions is None else self.action_encoder(actions)

        for seq_idx in range(1, seq_len):
            encoder_out_flat = encoder_out_flats[seq_idx]
            prev_z = z_s[seq_idx - 1]
            prev_u = None if u_s is None else u_s[seq_idx - 1]
            prev_w = w_s[-1]
            w, w_mu, w_log_var = self.get_w(encoder_out_flat)

            w_mus.append(w_mu)
            w_log_vars.append(w_log_var)
            w_s.append(w)
            z_s.append(self.get_next_z(prev_z, prev_w, prev_u))

        decoder_outs = [self.decoder(self.decoder_input(el).view((-1,) + self.decoder_input_size)) for el in z_s]
        return decoder_outs, z_s, w_s, w_mus, w_log_vars

    def simulate_next(self, z, u, device, w=None):
        if w is None:
            w = torch.randn((1, self.latent_dim)).to(device)
        z = self.get_next_z(z, w, u)
        return self.decoder(self.decoder_input(z).view((-1,) + self.decoder_input_size)), z


def vae_experiments():
    num_in_channels = 3
    im_size = (64, 64)
    input_size = (num_in_channels,) + im_size
    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)

    in_image = np.random.randint(256, size=input_size[1:] + (input_size[0],), dtype=np.uint8)
    in_image2 = np.random.randint(256, size=input_size[1:] + (input_size[0],), dtype=np.uint8)
    preprocessor = CNNPreProcessor()
    in_t = preprocessor.preprocess(in_image).to(device)
    in_t2 = preprocessor.preprocess(in_image2).to(device)

    in_t_b = torch.cat([in_t, in_t2])

    model = VariationalAutoEncoder.get_basic_vae(input_shape=input_size).to(device)

    out_t, mu_t, log_var_t = model(in_t)
    out_t_b, mu_t_b, log_var_t_b = model(in_t_b)

    def kl_loss(mu, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=1))

    mse_loss = F.mse_loss(in_t_b, out_t_b)

    kl_loss = kl_loss(mu_t_b, log_var_t_b)
    print("")


def sequence_experiments():
    num_in_channels = 3
    im_size = (64, 64)
    seq_len = 10
    batch_size = 4
    rnn_dim = 128

    num_actions = 3
    leave_one_out = True
    action_dim = num_actions - 1 if leave_one_out else num_actions

    input_size = (num_in_channels,) + im_size
    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)
    preprocessor = CNNPreProcessor()

    def generate_fake_sequence(s_len):
        return [np.random.randint(256, size=input_size[1:] + (input_size[0],), dtype=np.uint8) for _ in range(s_len)]

    def generate_fake_batch(b_len, pre, devc):
        ims = [np.random.randint(256, size=input_size[1:] + (input_size[0],), dtype=np.uint8) for _ in range(b_len)]
        return torch.cat([pre.preprocess(el) for el in ims]).to(devc)

    def generate_fake_action_sequence(s_len, n_actions, leave_one_out=True):
        from data import one_hot_encode
        return [one_hot_encode(np.random.randint(n_actions), n_actions, leave_one_out=leave_one_out) for _ in
                range(s_len)]

    fake_batch = []
    for _ in range(batch_size):
        seq = generate_fake_sequence(seq_len)
        seq_ts = [preprocessor.preprocess(el).to(device).unsqueeze(1) for el in seq]
        seq_ts = torch.cat(seq_ts)
        fake_batch.append(seq_ts)

    fake_action_batch = []
    for _ in range(batch_size):
        a_seq = generate_fake_action_sequence(seq_len, num_actions, leave_one_out=leave_one_out)
        a_seq_ts = [torch.from_numpy(el).unsqueeze(0).unsqueeze(0).type(torch.float32).to(device) for el in a_seq]
        a_seq_ts = torch.cat(a_seq_ts)
        fake_action_batch.append(a_seq_ts)

    action_in_t = torch.cat(fake_action_batch, dim=1)
    in_t = torch.cat(fake_batch, dim=1)

    fake_gt = generate_fake_batch(batch_size, preprocessor, device)

    model = SequencePredictor(input_size, build_basic_encoder(), build_basic_decoder(), rnn_dim,
                              action_dim=action_dim, bidirectional_rnn=True, rnn_layers=2).to(device)

    model_out = model(in_t, action_in_t)

    loss = F.mse_loss(fake_gt, model_out)

    loss.backward()

    print("")


if __name__ == "__main__":
    # vae_experiments()
    sequence_experiments()
