from torchinfo import summary
from tqdm import tqdm
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# First define the encoder, the encoder is a bert like transformer encoder. It inserts a [CLS] token at the beginning, followed by joint position, followed by k action sequences.


class ActionEncoder(nn.Module):
    def __init__(self, num_actions, joint_dim, lantent_dim=512):
        super().__init__()
        self.input_dim = joint_dim + num_actions * joint_dim
        self.lantent_dim = lantent_dim
        self.cls_token = nn.Parameter(
            torch.rand(
                1, lantent_dim
            )
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=lantent_dim, nhead=2
            ),  # joint pos and action seq
            num_layers=2,
        )

        self.joints_mlp = nn.Linear(joint_dim, lantent_dim)
        self.actions_mlp = nn.Linear(joint_dim, lantent_dim)

    def forward(self, joints, actions):
        joints = self.joints_mlp(joints)
        actions = self.actions_mlp(actions)
        # print(f'joints: {joints.shape}, actions: {actions.shape}, cls: {self.cls_token.shape}')
        X = torch.cat((self.cls_token, joints, actions), dim=0)
        # print(f'X: {X.shape}')
        X = self.transformer_encoder(X)
        return X[0].reshape(-1, self.lantent_dim)


class ActionDecoder(nn.Module):
    def __init__(self, lantent_dim, num_actions):
        super().__init__()
        self.latent_dim = lantent_dim
        self.num_actions = num_actions

        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.joints_mlp = nn.Linear(14, lantent_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=lantent_dim, nhead=4),
            num_layers=2,
        )

        def get_sinusoidal_embeddings(num_embeddings, embedding_dim):
            position = torch.arange(
                0, num_embeddings, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(
                0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
            embeddings = torch.zeros(num_embeddings, embedding_dim)
            embeddings[:, 0::2] = torch.sin(position * div_term)
            embeddings[:, 1::2] = torch.cos(position * div_term)
            return embeddings

        self.position_embeddings = nn.Parameter(
            get_sinusoidal_embeddings(num_actions, lantent_dim), requires_grad=False)

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=lantent_dim, nhead=4),
            num_layers=2,
        )
        self.mlp = nn.Linear(lantent_dim, 14)

    def forward(self, images, z, joint_pos):
        # images shape: [4, 3, 480, 640]
        features = self.resnet(images)
        features = rearrange(features, 'b (d c) -> (b d) c', c=self.latent_dim)
        # features shape: [1200, 512]

        joint_pos = self.joints_mlp(joint_pos)

        # Concatenate z and joint_pos with features
        combined_input = torch.cat((features, z, joint_pos), dim=0)
        # combined_input shape: [1202, 512]

        # Encode combined input using transformer encoder
        encoded_features = self.transformer_encoder(combined_input)
        # encoded_features shape: [1202, 512]

        # Decode using transformer decoder
        tgt = self.position_embeddings
        memory = encoded_features
        decoded_output = self.transformer_decoder(tgt, memory)
        # Map to final output dimension using MLP
        output = self.mlp(decoded_output)

        return output


class CVAE(nn.Module):
    """
    Convolutional Variational Autoencoder (CVAE) for action chunking.

    Attributes:
        encoder (ActionEncoder): Encoder network to process input joints and actions.
        mu_phi (nn.Linear): Linear layer to compute the mean of the latent variable.
        logvar (nn.Linear): Linear layer to compute the log variance of the latent variable.
        decoder (ActionDecoder): Decoder network to generate output action sequences.

    Methods:
        encode(joints, actions):
            Encodes the input joints and actions into latent space.

        reparametrize(mu_phi, logvar):
            Reparameterizes the latent variable using the mean and log variance.

        decode(images, z, joint_pos):
            Decodes the latent variable to generate output action sequences.

        forward(current_joints, future_actions, images):
            Forward pass through the CVAE model.

        loss_fn(predict, target, mu_phi, logvar):
            Computes the loss function, including KL divergence and reconstruction loss.
    """

    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder = ActionEncoder(
            num_actions=4, joint_dim=14, lantent_dim=128)
        self.mu_phi = nn.Linear(128, 128)
        self.logvar = nn.Linear(128, 128)
        # output action seq directly
        self.decoder = ActionDecoder(lantent_dim=128, num_actions=4)

    def encode(self, joints, actions):
        z = self.encoder(joints, actions)
        return self.mu_phi(z), self.logvar(z)

    def reparametrize(self, mu_phi, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).detach().to(device)
        return mu_phi + eps * std

    def decode(self, images, z, joint_pos):
        return self.decoder(images, z, joint_pos)

    def forward(self, current_joints, future_actions, images):
        mu_phi, logvar = self.encode(current_joints, future_actions)
        z = self.reparametrize(mu_phi, logvar)
        return self.decode(images, z, current_joints), mu_phi, logvar

    def loss_fn(self, predict, target, mu_phi, logvar):
        # KL divergence: Sum over latent dimensions, average over batch
        kl_divergence = -0.5 * \
            torch.sum(1 + logvar - mu_phi.pow(2) - logvar.exp(), dim=1).mean()
        # Reconstruction loss: Mean squared error
        # Sum over features, average over batch
        reconstruction = F.mse_loss(predict, target, reduction='mean')

        # Total loss
        return kl_divergence + reconstruction


if __name__ == '__main__':
    model = CVAE().to(device)
    with torch.no_grad():
        pred_actions, mu_phi, logvar = model(
            torch.randn(4, 14).to(device), torch.randn(4, 14).to(device), torch.randn(4, 3, 480, 640).to(device))
        print(pred_actions.shape, mu_phi.shape, logvar.shape)
        print(model.loss_fn(torch.randn(4, 14).to(device), torch.randn(
            4, 14).to(device), torch.randn(1, 128).to(device), torch.randn(1, 128).to(device)))
