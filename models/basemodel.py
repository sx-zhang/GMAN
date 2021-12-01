from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.net_util import norm_col_init, weights_init

from .model_io import ModelOutput
import numpy as np
import scipy.io as scio
from torch.autograd import Variable

def generator_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):

    def __init__(self, decoder_layer_sizes,latent_size):

        super(Generator, self).__init__()

        layer_sizes = decoder_layer_sizes
        latent_size = latent_size
        input_size = latent_size * 2
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(generator_weights_init)

    def _forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc3(x1))
        self.out = x1
        return x

    def forward(self, z, a1=None, c=None, feedback_layers=None):
        # if feedback_layers is None:  # zsx removed
        #     return self._forward(z, c)
        # else:
        #     z = torch.cat((z, c), dim=-1)
        #     x1 = self.lrelu(self.fc1(z))
        #     feedback_out = x1 + a1*feedback_layers
        #     x = self.sigmoid(self.fc3(feedback_out))
        #     return x
        return self._forward(z, c)


class BaseModel(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        target_embedding_sz = args.glove_dim
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(BaseModel, self).__init__()

        # att detector embedding
        self.att_detection_linear_1 = nn.Linear(2, 49)
        self.att_detection_linear_2 = nn.Linear(49, 49)
        self.att_detection_linear_3 = nn.Linear(49, 49)
        self.att_detection_linear_4 = nn.Linear(49, 49)
        self.att_detection_linear_5 = nn.Linear(49, 49)

        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.embed_glove = nn.Linear(target_embedding_sz, 64)
        self.embed_action = nn.Linear(action_space, 10)
        self.embed_fake = nn.Linear(2048, 32 * 7 * 7)

        pointwise_in_channels = 138 + 45 + 32

        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        lstm_input_sz = 7 * 7 * 64

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
        num_outputs = action_space
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_space)

        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.attributes_dictionary = scio.loadmat('./data/attributes/40_attribute.mat')
        self.att_episode = None
        # generator
        decoder_layer_sizes = [4096, 2048]
        latent_size = 45
        generator = Generator(decoder_layer_sizes, latent_size)
        generator.load_state_dict(torch.load('./pretrained_models/netG.pth', map_location=lambda storage, loc: storage))
        # generator.load_state_dict(torch.load('./pretrained_models/netG.pth'))
        self.generator = generator
        for param in self.generator.parameters():
            if args.G_grad:
                continue
            else:
                param.requires_grad = False
        self.fake_img = None

    def embedding(self, state, target, action_probs, params, att, att_in_view):
        # noise = torch.FloatTensor(1, 45)
        # noise.normal_(0, 1)
        noise = torch.zeros(1, 45)  # fix noise
        z = Variable(noise).to(torch.device(target.device))
        att_in_view_cuda = torch.tensor(att_in_view).to(torch.device(target.device))
        all_att = torch.cat((att_in_view_cuda, att), dim=0).float()

        action_embedding_input = action_probs

        if params is None:
            fake = self.generator(z, c=att.view(1, 45))
            self.fake_img = F.avg_pool2d(fake.view(1, 512, 2, 2), 2).view(1, 512)
            fake_embedding = F.relu(self.embed_fake(fake.squeeze()))
            fake_reshaped = fake_embedding.view(1, 32, 7, 7)

            glove_embedding = F.relu(self.embed_glove(target))
            glove_reshaped = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)

            att_embedding = F.relu(self.att_detection_linear_1(all_att.t()))
            att_embedding = F.relu(self.att_detection_linear_2(att_embedding))
            att_embedding = F.relu(self.att_detection_linear_3(att_embedding))
            att_embedding = F.relu(self.att_detection_linear_4(att_embedding))
            att_embedding = F.relu(self.att_detection_linear_5(att_embedding))
            att_reshaped = att_embedding.view(1, 45, 7, 7)

            action_embedding = F.relu(self.embed_action(action_embedding_input))
            action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

            image_embedding = F.relu(self.conv1(state))
            x = self.dropout(image_embedding)
            # x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)
            x = torch.cat((x, glove_reshaped, att_reshaped, action_reshaped, fake_reshaped), dim=1)
            x = F.relu(self.pointwise(x))
            x = self.dropout(x)
            out = x.view(x.size(0), -1)

        else:
            generator_dict = {}
            for k in params:
                if k.split('.')[0] == 'generator':
                    nk = k[10:]
                    generator_dict[nk] = params[k]

            self.generator.load_state_dict(generator_dict)
            fake = self.generator(z, c=att.view(1, 45))  # seems something wrong
            self.fake_img = F.avg_pool2d(fake.view(1, 512, 2, 2), 2).view(1, 512)

            fake_embedding = F.relu(
                F.linear(
                    fake.squeeze(),
                    weight=params["embed_fake.weight"],
                    bias=params["embed_fake.bias"],
                )
            )
            fake_reshaped = fake_embedding.view(1, 32, 7, 7)

            glove_embedding = F.relu(
                F.linear(
                    target,
                    weight=params["embed_glove.weight"],
                    bias=params["embed_glove.bias"],
                )
            )

            glove_reshaped = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)

            att_embedding = F.relu(
                F.linear(
                    all_att.t(),
                    weight=params["att_detection_linear_1.weight"],
                    bias=params["att_detection_linear_1.bias"],
                )
            )

            att_embedding = F.relu(
                F.linear(
                    att_embedding,
                    weight=params["att_detection_linear_2.weight"],
                    bias=params["att_detection_linear_2.bias"],
                )
            )

            att_embedding = F.relu(
                F.linear(
                    att_embedding,
                    weight=params["att_detection_linear_3.weight"],
                    bias=params["att_detection_linear_3.bias"],
                )
            )

            att_embedding = F.relu(
                F.linear(
                    att_embedding,
                    weight=params["att_detection_linear_4.weight"],
                    bias=params["att_detection_linear_4.bias"],
                )
            )

            att_embedding = F.relu(
                F.linear(
                    att_embedding,
                    weight=params["att_detection_linear_5.weight"],
                    bias=params["att_detection_linear_5.bias"],
                )
            )
            att_reshaped = att_embedding.view(1, 45, 7, 7)

            action_embedding = F.relu(
                F.linear(
                    action_embedding_input,
                    weight=params["embed_action.weight"],
                    bias=params["embed_action.bias"],
                )
            )
            action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

            image_embedding = F.relu(
                F.conv2d(
                    state, weight=params["conv1.weight"], bias=params["conv1.bias"]
                )
            )
            x = self.dropout(image_embedding)
            # x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)
            x = torch.cat((x, glove_reshaped, att_reshaped, action_reshaped, fake_reshaped), dim=1)

            x = F.relu(
                F.conv2d(
                    x, weight=params["pointwise.weight"], bias=params["pointwise.bias"]
                )
            )
            x = self.dropout(x)
            out = x.view(x.size(0), -1)

        return out, image_embedding

    def a3clstm(self, embedding, prev_hidden, params):
        if params is None:
            hx, cx = self.lstm(embedding, prev_hidden)
            x = hx
            actor_out = self.actor_linear(x)
            critic_out = self.critic_linear(x)

        else:
            # hx, cx = self._backend.LSTMCell(
            #     embedding,
            #     prev_hidden,
            #     params["lstm.weight_ih"],
            #     params["lstm.weight_hh"],
            #     params["lstm.bias_ih"],
            #     params["lstm.bias_hh"],
            # )

            # Change for pytorch 1.01
            # hx, cx = nn._VF.lstm_cell(
            #     embedding,
            #     prev_hidden,
            #     params["lstm.weight_ih"],
            #     params["lstm.weight_hh"],
            #     params["lstm.bias_ih"],
            #     params["lstm.bias_hh"],
            # )

            # pytorch 1.7
            hx, cx = torch._VF.lstm_cell(
                embedding,
                prev_hidden,
                params["lstm.weight_ih"],
                params["lstm.weight_hh"],
                params["lstm.bias_ih"],
                params["lstm.bias_hh"],
            )

            x = hx

            critic_out = F.linear(
                x,
                weight=params["critic_linear.weight"],
                bias=params["critic_linear.bias"],
            )
            actor_out = F.linear(
                x,
                weight=params["actor_linear.weight"],
                bias=params["actor_linear.bias"],
            )

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):

        state = model_input.state
        (hx, cx) = model_input.hidden

        att_in_view = model_input.att_in_view
        target_object = model_input.target_object
        att_episode = torch.from_numpy(self.attributes_dictionary[target_object].astype(np.float32)).to(
            torch.device(hx.device))

        target = model_input.target_class_embedding
        action_probs = model_input.action_probs
        params = model_options.params

        x, image_embedding = self.embedding(state, target, action_probs, params, att_episode, att_in_view)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx), params)
        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
            fake_img=self.fake_img,
        )
