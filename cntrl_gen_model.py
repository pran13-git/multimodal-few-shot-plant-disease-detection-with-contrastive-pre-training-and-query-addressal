import random
import sys
import numpy as np
import os
import torch 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import texar.torch as tx
from texar.torch.modules import WordEmbedder,  UnidirectionalRNNEncoder, MLPTransformConnector, AttentionRNNDecoder, beam_search_decode,\
      GumbelSoftmaxEmbeddingHelper, UnidirectionalRNNClassifier
from texar.torch.core import get_train_op
from texar.torch.utils import collect_trainable_variables, get_batch_size

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class CtrlGenModel(object):
    """
    Generates Data
    * Input: text and labels
    * Output: text that not only matches the input text but also satisfies certain control criteria

    """

    def __init__(self, inputs, vocab, lambda_ae, gamma, lambda_D,lambda_diversity, hparams=None):

        self._hparams = tx.HParams(hparams, None) #* maintains hyperparameters for configuring Texar modules
        self._build_model(inputs, vocab, lambda_ae, gamma, lambda_D,lambda_diversity)

    def _build_model(self, inputs, vocab, lambda_ae, gamma, lambda_D,lambda_diversity):
        """
        Builds the model
        """

        """
            * SECTION 1
            * Get embeddings and final layer representation
            * Get generated sequences by feeding this to a decoder
        """
        embedder = WordEmbedder(vocab_size=vocab.size,hparams=self._hparams.embedder)
        # rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

        # defining initial state
        encoder = UnidirectionalRNNEncoder(hparams=self._hparams.encoder)
        # initial_state=rnn_cell.zero_state(64, dtype=tf.float32)

        # text_ids for encoder, with BOS removed
        enc_text_ids = inputs['text_ids'][:, 1:]

        # *  final hidden state of the UnidirectionalRNNEncoder obtained during the encoding process
        # *  Extracted part of the hidden state is often used as a representation or embedding of the input sequence 
        # *  here all rows, and columns from self._hparams.dim_c are taken
        enc_outputs, final_state = encoder(embedder(enc_text_ids), sequence_length=inputs['length'] - 1)
        z = final_state[:, self._hparams.dim_c:]

        # Encodes label
        label_connector = MLPTransformConnector(self._hparams.dim_c)

        # Gets the sentence representation: h = (c, z)
        labels = inputs['labels'].view(-1, 1).float() # * column vector of float values.
        c = label_connector(labels) 
        c_ = label_connector(1 - labels)
        h = torch.cat([c, z], dim=1)
        h_ = torch.cat([c_, z], dim=1)


        # Teacher-force decoding and the auto-encoding loss for G
        # * This decoder is designed to generate sequences while attending to specific parts of the input sequence 
        decoder = AttentionRNNDecoder(
            memory=enc_outputs,
            memory_sequence_length=inputs['length'] - 1,
            cell_input_fn=lambda inputs, attention: inputs,
            vocab_size=vocab.size,
            hparams=self._hparams.decoder)

        connector = MLPTransformConnector(decoder.state_size)

        # g_outputs shape = (64, 17)
        # * generated sequences produced by the decoder 
        g_outputs, _, _ = decoder(initial_state=connector(h), 
                                  inputs=inputs['text_ids'],
                                  embedding=embedder, 
                                  sequence_length=inputs['length'] - 1)

        # sequence_sparse_softmax_cross_entropy <---> tf.nn.softmax_cross_entropy_with_logits_v2
        #    1. calculate y_hat_softmax: softmax to logits(y_hat)
        #    2. compute cross entropy---> y*tf.log(y_hat_softmax)
        #    3. Sum over different class for an instance

        #################### TEST TEST TEST ###########################################

        """
            * SECTION 2
            * calculates a diversity loss based on the softmax probabilities and one-hot encoded labels
        """
        labels = torch.tensor([[1, 2, 0], [1, 2, 1]])
        test_labels = F.one_hot(labels, num_classes=3).float()
        test_logits = torch.tensor([[[0.1, 0.2, 0.3], [0.1, 0.1, 0.6], [1, 2.1, 5]],
                                    [[2., 0.2, 4.], [4.5, 0.1, 2.5], [1, 1, 5]]])

        test_softmax_logits = F.softmax(test_logits, dim=-1)
        test_diff = test_softmax_logits - test_labels

        test_diff_clipped = torch.clamp(test_diff, 0 + 1e-8, 1)
        test_diff_clipped_minibatch = torch.mean(test_diff_clipped, dim=0)

        test_entropy_minibatch = -1.0 * torch.sum(test_diff_clipped_minibatch * torch.log(test_diff_clipped_minibatch))
        test_loss_diversity = test_entropy_minibatch.item()  # Use .item() to get a Python scalar


        ############################# Diversity LOSS ######################################
        """
        ? Normal diversity loss (not test?)
        """

        one_hot_labels = F.one_hot(inputs['text_ids'][:, 1:], num_classes=vocab.size).float()

        softmax_logits = F.softmax(g_outputs.logits, dim=-1)
        diff = softmax_logits - one_hot_labels

        diff_clipped = torch.clamp(diff, 0 + 1e-8, 1)
        diff_clipped_minibatch = torch.mean(diff_clipped, dim=0)

        entropy_minibatch = -1.0 * torch.sum(diff_clipped_minibatch * torch.log(diff_clipped_minibatch))
        loss_diversity = entropy_minibatch.item()  # Use .item() to get a Python scalar

        #########################################################################
        
        """
         * SECTION 3
         * Calculates and stores the autoencoder loss, ground truth labels, and autoencoder logits
        """
        loss_g_ae = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=inputs['text_ids'][:, 1:],
            logits=g_outputs.logits,
            sequence_length=inputs['length'] - 1,
            average_across_timesteps=True,
            sum_over_timesteps=False)

        self.test1 = loss_g_ae

        self.input_labels_shape = inputs['text_ids'][:, 1:]
        self.my_g_ouputslogits = g_outputs.logits


        # Gumbel-softmax decoding, used in training
        start_tokens = torch.ones_like(torch.tensor(inputs['labels'])) * vocab.bos_token_id

        end_token = vocab.eos_token_id
        gumbel_helper = GumbelSoftmaxEmbeddingHelper(
            embedder.embedding, start_tokens, end_token, gamma)
        soft_outputs_, _, soft_length_, = decoder(
            helper=gumbel_helper, initial_state=connector(h_))

        # Greedy decoding, used in evaluation
        outputs_, _, length_ = decoder(
            decoding_strategy='infer_greedy', initial_state=connector(h_),
            embedding=embedder, start_tokens=start_tokens, end_token=end_token)

        # Creates discriminator

        classifier = UnidirectionalRNNClassifier(hparams=self._hparams.classifier)
        clas_embedder = WordEmbedder(vocab_size=vocab.size,
                                     hparams=self._hparams.embedder)

        # Assuming classifier is a PyTorch module
        _, clas_logits, clas_preds = classifier(
            inputs=clas_embedder(ids=inputs['text_ids'][:, 1:]),
            sequence_length=inputs['length'] - 1)

        # Classification loss for the classifier
        loss_d_clas = F.binary_cross_entropy_with_logits(
            clas_logits, torch.tensor(inputs['labels']).float())

        prob = torch.sigmoid(clas_logits)

        loss_d_clas = loss_d_clas.item()  # Use .item() to get a Python scalar
        accu_d = (clas_preds == inputs['labels']).float().mean().item()  # Use .item() to get a Python scalar

        # Classification loss for the generator, based on soft samples
        _, soft_logits, soft_preds = classifier(
            inputs=clas_embedder(soft_ids=soft_outputs_.sample_id),
            sequence_length=soft_length_)

        # Assuming the soft targets are the complementary labels
        loss_g_clas = F.binary_cross_entropy_with_logits(
            soft_logits, 1 - torch.tensor(inputs['labels']).float())

        loss_g_clas = loss_g_clas.item()  # Use .item() to get a Python scalar

        # Accuracy on soft samples, for training progress monitoring
        accu_g = tx.evals.accuracy(labels=1 - inputs['labels'], preds=soft_preds)

        # Accuracy on greedy-decoded samples, for training progress monitoring
        """
        * beam search to generate sequences and evaluates the accuracy of the generated sequences using a classifier
        """
        beam_outputs, _, _, = beam_search_decode(
            decoder_or_cell=decoder,
            embedding=embedder,
            start_tokens=start_tokens,
            end_token=end_token,
            beam_width=3,
            initial_state=connector(h_),
            max_decoding_length=21)

        _,_, gdy_preds = classifier(
            inputs=clas_embedder(ids=outputs_.sample_id),
            sequence_length=length_)
        accu_g_gdy = tx.evals.accuracy(
            labels=1 - inputs['labels'], preds=gdy_preds)

        # Aggregates losses
        loss_g = (lambda_ae * loss_g_ae) + (lambda_D * loss_g_clas) +(lambda_diversity * loss_diversity)
        loss_d = loss_d_clas

        # Summaries for losses
        loss_g_ae_summary = loss_g_ae  # PyTorch doesn't require explicit summary creation
        loss_diversity_summary = loss_diversity  # PyTorch doesn't require explicit summary creation
        loss_g_clas_summary = loss_g_clas  # PyTorch doesn't require explicit summary creation

        # Assuming you have a TensorBoard SummaryWriter created
        writer = SummaryWriter()

        # Add the PyTorch tensors to TensorBoard
        writer.add_scalar('loss_g_ae_summary', loss_g_ae_summary)#, global_step=your_global_step)
        writer.add_scalar('loss_diversity_summary', loss_diversity_summary)#, global_step=your_global_step)
        writer.add_scalar('loss_g_clas_summary', loss_g_clas_summary) #, global_step=your_global_step)

        # Creates optimizers IMPORTANT CHECK
        g_vars = collect_trainable_variables(
            [embedder, encoder, label_connector, connector, decoder])
        d_vars = collect_trainable_variables([clas_embedder, classifier])

        train_op_g = get_train_op(
            loss_g, g_vars, hparams=self._hparams.opt)
        train_op_g_ae = get_train_op(
            loss_g_ae, g_vars, hparams=self._hparams.opt)
        train_op_d = get_train_op(
            loss_d, d_vars, hparams=self._hparams.opt)

        # Interface tensors
        self.losses = {
            "loss_g": loss_g,
            "loss_g_ae": loss_g_ae,
            "loss_diversity": loss_diversity,
            "loss_g_clas": loss_g_clas,
            "loss_d": loss_d_clas
        }

        self.metrics = {
            "accu_d": accu_d,
            "accu_g": accu_g,
            "accu_g_gdy": accu_g_gdy
        }

        self.train_ops = {
            "train_op_g": train_op_g,
            "train_op_g_ae": train_op_g_ae,
            "train_op_d": train_op_d
        }

        self.samples = {
            "original": inputs['text_ids'],
            "original_labels": inputs['labels'],
            "transferred": outputs_.sample_id,
            "beam_transferred": beam_outputs.predicted_ids,
            "soft_transferred": soft_outputs_.sample_id
        }

        self.summaries = {
            "loss_g_ae_summary": loss_g_ae_summary,
            "loss_g_clas_summary": loss_g_clas_summary,
            "loss_diversity_summary": loss_diversity_summary,
        }

        self.fetches_train_g = {
            "loss_g": self.train_ops["train_op_g"],
            "loss_g_ae": self.losses["loss_g_ae"],
            "loss_diversity": self.losses["loss_diversity"],
            "loss_g_clas": self.losses["loss_g_clas"],
            "accu_g": self.metrics["accu_g"],
            "accu_g_gdy": self.metrics["accu_g_gdy"],
            "loss_g_ae_summary": self.summaries["loss_g_ae_summary"],
            "loss_g_clas_summary": self.summaries["loss_g_clas_summary"],
            "batch_size": get_batch_size(inputs['text_ids']),
            # "loss_diversity_summary ": self.summaries["loss_diversity_summary"],
        }

        self.fetches_train_d = {
            "loss_d": self.train_ops["train_op_d"],
            "accu_d": self.metrics["accu_d"],
            "y_prob": prob,
            "y_pred": clas_preds,
            "y_true": inputs['labels'],
            "sentences": inputs['text_ids']
        }

        self.fetches_dev_test_d = {
            "y_prob": prob,
            "y_pred": clas_preds,
            "y_true": inputs['labels'],
            "sentences": inputs['text_ids'],

            "batch_size": get_batch_size(inputs['text_ids']),
            "loss_d": self.losses['loss_d'],
            "accu_d": self.metrics["accu_d"],
        }

        fetches_eval = {"batch_size": get_batch_size(inputs['text_ids'])}
        fetches_eval.update(self.losses)
        fetches_eval.update(self.metrics)
        fetches_eval.update(self.samples)
        self.fetches_eval = fetches_eval