import tensorflow.contrib.slim as slim
import tensorflow as tf
import tensorflow.contrib.distributions as ds

# from base_models import BaseModelWithDefaultCritic
from base_models import (BaseModel, CriticModelMixin, StochasticActorModelMixin,
                         PointEstimateActorModelMixin, LinearModelMixin,
                         StochasticLinearActorModelMixin, ProperlyShapedPointEstimateModelMixin)
# from warp_utils import ErrorVectorCreator
from warp_utils import ErrorVectorCreator
from lambdarank_utils import my_lambdarank


class OldMultiDAE(CriticModelMixin, PointEstimateActorModelMixin, BaseModel):

    """
    This implements the MultiDAE from https://github.com/dawenl/vae_cf
    """

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            anneal_cap=0.2,
            epochs_to_anneal_over=50,
            batch_size=500,
            evaluation_metric='NDCG',
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            actor_reg_loss_scaler=1e-4,  #The default in training is 1e-4 so I'll leave it like that.
            ac_reg_loss_scaler=0.0,  # We'll increase it as need be.
            **kwargs):
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        # NOTE: Check the KL term later.
        # NOTE: This will give the wrong number for validation stuff, because batch_of_users isn't the right output!.

        self.actor_error_mask = tf.identity(self.batch_of_users)

        log_softmax = tf.nn.log_softmax(self.prediction)
        actor_error = -tf.reduce_sum(log_softmax * self.actor_error_mask, axis=-1)

        # This way, KL isn't factored into the critic at all. Which is probably what we want, although the AC should have it.
        # mean_actor_error = tf.reduce_mean(actor_error) + (
        #     self.kl_loss_scaler * self.actor_regularization_loss)
        mean_actor_error = tf.reduce_mean(actor_error) + (
            self.actor_reg_loss_scaler * self.actor_regularization_loss)

        self.actor_error = actor_error
        self.mean_actor_error = mean_actor_error

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)

        
        
class MultiVAE(CriticModelMixin, StochasticActorModelMixin, BaseModel):

    """This implements the MultaVAE from https://github.com/dawenl/vae_cf"""

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            anneal_cap=0.2,
            epochs_to_anneal_over=50,
            batch_size=500,
            evaluation_metric='NDCG',
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            ac_reg_loss_scaler=0.0,  # We'll increase it as need be.
            omit_num_seen_from_critic=False,
            omit_num_unseen_from_critic=False,
            **kwargs):
        """
        I'll do a better job about defining the inputs here.
        """
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

        if hasattr(self, 'superclass_stuff'):
            self.superclass_stuff()


    def construct_actor_error(self):
        # NOTE: Check the KL term later.
        # NOTE: This will give the wrong number for validation stuff, because batch_of_users isn't the right output!.

        self.actor_error_mask = tf.identity(self.batch_of_users)

        log_softmax = tf.nn.log_softmax(self.prediction)
        actor_error = -tf.reduce_sum(log_softmax * self.actor_error_mask, axis=-1)

        # This way, KL isn't factored into the critic at all. Which is probably what we want, although the AC should have it.
        mean_actor_error = tf.reduce_mean(actor_error) + (
            self.kl_loss_scaler * self.actor_regularization_loss)

        self.actor_error = actor_error
        self.mean_actor_error = mean_actor_error

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)

        # tf.summary.scalar("NDCG@200", tf.reduce_mean(self.true_ndcg_at_200))
        # tf.summary.scalar("NDCG@50", tf.reduce_mean(self.true_ndcg_at_50))
        # tf.summary.scalar("NDCG@20", tf.reduce_mean(self.true_ndcg_at_20))
        # tf.summary.scalar("NDCG@5", tf.reduce_mean(self.true_ndcg_at_5))
        
        
class MultiVAEWithUserinfo(MultiVAE):
    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            anneal_cap=0.2,
            epochs_to_anneal_over=50,
            batch_size=500,
            evaluation_metric='NDCG',
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            ac_reg_loss_scaler=0.0,  # We'll increase it as need be.
            omit_num_seen_from_critic=False,
            omit_num_unseen_from_critic=False,
            **kwargs):
        """
        I'll do a better job about defining the inputs here.
        """
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

        if hasattr(self, 'superclass_stuff'):
            self.superclass_stuff()
            
    def construct_actor_error(self):
        # NOTE: Check the KL term later.
        # NOTE: This will give the wrong number for validation stuff, because batch_of_users isn't the right output!.

        self.actor_error_mask = tf.identity(self.batch_of_users[:,:self.input_dim])

        log_softmax = tf.nn.log_softmax(self.prediction)
        actor_error = -tf.reduce_sum(log_softmax * self.actor_error_mask, axis=-1)

        # This way, KL isn't factored into the critic at all. Which is probably what we want, although the AC should have it.
        mean_actor_error = tf.reduce_mean(actor_error) + (
            self.kl_loss_scaler * self.actor_regularization_loss)

        self.actor_error = actor_error
        self.mean_actor_error = mean_actor_error
    
    def construct_masked_inputs(self):
        masker = ds.Bernoulli(probs=self.keep_prob_ph, dtype=tf.float32)
        mask_shape = [self.batch_size, self.input_dim]
        mask = masker.sample(sample_shape=mask_shape)
        reverse_mask = (1 - mask)  #Only leaves the things that aren't in the original input.
        network_input = (self.batch_of_users[:,:self.input_dim] * mask)
        remaining_input = (self.batch_of_users[:,:self.input_dim] * reverse_mask)

        number_of_good_items = tf.reduce_sum(self.batch_of_users[:,:self.input_dim], axis=-1)
        number_of_unseen_items = tf.reduce_sum(remaining_input, axis=-1)
        number_of_seen_items = tf.reduce_sum(network_input, axis=-1)

        self.mask = mask
        self.network_input = tf.concat([network_input,self.batch_of_users[:,self.input_dim:]],1)      # masked input (input for actors)
        self.remaining_input = remaining_input  # reverse masked input
        self.number_of_good_items = number_of_good_items    # feature H0
        self.number_of_unseen_items = number_of_unseen_items    # feature H1
        self.number_of_seen_items = number_of_seen_items

    def _create_normalized_network_input(self):
        normalized_network_input = tf.nn.l2_normalize(self.batch_of_users[:,:self.input_dim], 1)
        normalized_network_input = (normalized_network_input * self.mask) / self.keep_prob_ph
        normalized_network_input = tf.concat([normalized_network_input,self.batch_of_users[:,self.input_dim:]],1)
        self.normalized_network_input = normalized_network_input
    
    def _create_critic_input_vector(self):
        critic_inputs = []
        if not getattr(self, 'omit_num_seen_from_critic', False):
            print("NOT OMITTING NUM SEEN")
            critic_inputs.append(self.number_of_seen_items) # get |H_0|
        else:
            print("OMITTING NUM SEEN")
        if not getattr(self, 'omit_num_unseen_from_critic', False):
            print('NOT OMITTING NUM UNSEEN')
            critic_inputs.append(self.number_of_unseen_items)   # get |H_1|
        else:
            print("OMITTING NUM UNSEEN")
        
        print("Always doing actor error, of course.")
        critic_inputs.append(self.actor_error)  # get L_E
        
        unnormalized_ac_input = critic_inputs[0]
        
        for i in range(1, len(critic_inputs)):
            unnormalized_ac_input = tf.concat([unnormalized_ac_input,critic_inputs[i]],0)
        unnormalized_ac_input = tf.reshape(unnormalized_ac_input,[critic_inputs[0].shape[0],-1])
        
        unnormalized_ac_input = tf.concat([unnormalized_ac_input,self.batch_of_users[:,self.input_dim:]], 1)

        #unnormalized_ac_input = tf.stack(critic_inputs, axis=-1) # change the 3 batch_size_dimensional vectors to batch_size number of 3_dimensional vectors

        self.ac_input = tf.contrib.layers.batch_norm(
            unnormalized_ac_input,
            is_training=self.train_batch_norm,
            trainable=False,  #Don't know what scale is really, but it says if relu next, don't use.
            renorm=True,  #Not sure. But makes it use closer stats for training and testing.
        )
    def create_validation_ops(self):
        # This is going to be a sexy graph replace.
        # network-input is the same. The actor-error-mask is different, although honestly, it doesn't matter
        # what actor-error is for validation.
        vad_true_ndcg, vad_critic_error, vad_prediction = \
            tf.contrib.graph_editor.graph_replace(
                [self.true_ndcg, self.critic_error, self.prediction],
                {
                    self.remaining_input : self.heldout_batch,
                    self.actor_error_mask : (self.batch_of_users[:,:self.input_dim] + self.heldout_batch[:,:self.input_dim])
                })

        # vad_true_ndcg, vad_true_ap, vad_true_recall, vad_critic_error, vad_prediction = \
        #     tf.contrib.graph_editor.graph_replace(
        #         [self.true_ndcg, self.true_ap, self.true_recall, self.critic_error, self.prediction],
        #         {
        #             self.remaining_input : self.heldout_batch,
        #             self.actor_error_mask : (self.batch_of_users + self.heldout_batch)
        #         })

        # vad_true_ndcg_at_200, vad_true_ndcg_at_50, vad_true_ndcg_at_20, vad_true_ndcg_at_5 = \
        #     tf.contrib.graph_editor.graph_replace(
        #         [self.true_ndcg_at_200, self.true_ndcg_at_50, self.true_ndcg_at_20, self.true_ndcg_at_5],
        #         {
        #             self.remaining_input : self.heldout_batch,
        #             self.actor_error_mask : (self.batch_of_users + self.heldout_batch)
        #         })

        vad_actor_error, vad_critic_output = \
            tf.contrib.graph_editor.graph_replace(
                [self.actor_error, self.critic_output],
                {
                    self.remaining_input : self.heldout_batch,
                    self.actor_error_mask : (self.batch_of_users[:,:self.input_dim] + self.heldout_batch[:,:self.input_dim])
                })

        vad_true_evaluation_metric = \
            tf.contrib.graph_editor.graph_replace(
                self.true_evaluation_metric,
                {
                    self.remaining_input : self.heldout_batch,
                    self.actor_error_mask : (self.batch_of_users[:,:self.input_dim] + self.heldout_batch[:,:self.input_dim])
                })
        
        self.vad_true_evaluation_metric = vad_true_evaluation_metric

        self.vad_true_ndcg = vad_true_ndcg
        # self.vad_true_ap = vad_true_ap
        # self.vad_true_recall = vad_true_recall

        # self.vad_true_ndcg_at_200 = vad_true_ndcg_at_200
        # self.vad_true_ndcg_at_50 = vad_true_ndcg_at_50
        # self.vad_true_ndcg_at_20 = vad_true_ndcg_at_20
        # self.vad_true_ndcg_at_5 = vad_true_ndcg_at_5
        

        self.vad_critic_error = vad_critic_error
        self.vad_prediction = vad_prediction
        self.vad_actor_error = vad_actor_error
        self.vad_critic_output = vad_critic_output

    def construct_critic_error(self):
        true_dcg = self._return_unnormalized_dcg_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input[:,:self.input_dim],
            input_batch=self.network_input[:,:self.input_dim])

        true_ndcg = self._return_ndcg_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input[:,:self.input_dim],
            input_batch=self.network_input[:,:self.input_dim])

        true_ndcg_at_200 = self._return_ndcg_at_200_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input[:,:self.input_dim],
            input_batch=self.network_input[:,:self.input_dim])

        true_ndcg_at_5 = self._return_ndcg_at_5_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input[:,:self.input_dim],
            input_batch=self.network_input[:,:self.input_dim])

        true_ndcg_at_3 = self._return_ndcg_at_3_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input[:,:self.input_dim],
            input_batch=self.network_input[:,:self.input_dim])

        true_ndcg_at_1 = self._return_ndcg_at_1_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input[:,:self.input_dim],
            input_batch=self.network_input[:,:self.input_dim])

        # true_ap = self._return_ap_given_args(
        #     our_outputs=self.prediction,
        #     true_outputs=self.remaining_input,
        #     input_batch=self.network_input)
        
        true_recall = self._return_recall_given_args(
            our_outputs=self.prediction,
            true_outputs=self.remaining_input[:,:self.input_dim],
            input_batch=self.network_input[:,:self.input_dim])

        if self.evaluation_metric == 'NDCG':
            print("Evaluating with NDCG")
            evaluation_metric = true_ndcg

        elif self.evaluation_metric == 'NDCG_AT_200':
            evaluation_metric = true_ndcg_at_200
        elif self.evaluation_metric == 'NDCG_AT_5':
            evaluation_metric = true_ndcg_at_5
        elif self.evaluation_metric == 'NDCG_AT_3':
            evaluation_metric = true_ndcg_at_3
        elif self.evaluation_metric == 'NDCG_AT_1':
            evaluation_metric = true_ndcg_at_1

        elif self.evaluation_metric == 'DCG':
            evaluation_metric = true_dcg

        # elif self.evaluation_metric == 'AP':
        #     print("Evaluating with AP")
        #     evaluation_metric = true_ap
        # elif self.evaluation_metric == 'RECALL':
        #     evaluation_metric = true_recall
        # elif self.evaluation_metric == 'NDCG_AT_200':
        #     evaluation_metric = true_ndcg_at_200
        # elif self.evaluation_metric == 'NDCG_AT_50':
        #     evaluation_metric = true_ndcg_at_50
        # elif self.evaluation_metric == 'NDCG_AT_20':
        #     evaluation_metric = true_ndcg_at_20
        # elif self.evaluation_metric == 'NDCG_AT_5':
        #     evaluation_metric = true_ndcg_at_5
        else:
            raise ValueError("evaluation_metric must be one of NDCG, AP, RECALL, or one of the NDCG ones. Instead got {}".format(
                self.evaluation_metric))

        critic_error = (evaluation_metric - self.critic_output)**2
        self._build_critic_reg()
        mean_critic_error = tf.reduce_mean(critic_error) + self.critic_regularization_loss

        self.true_dcg = true_dcg
        self.true_ndcg = true_ndcg
        # self.true_ap = true_ap
        self.true_recall = true_recall

        self.true_ndcg_at_200 = true_ndcg_at_200
        # self.true_ndcg_at_50 = true_ndcg_at_50
        # self.true_ndcg_at_20 = true_ndcg_at_20
        self.true_ndcg_at_5 = true_ndcg_at_5
        self.true_ndcg_at_3 = true_ndcg_at_3
        self.true_ndcg_at_1 = true_ndcg_at_1

        self.critic_error = critic_error
        self.mean_critic_error = mean_critic_error
        self.true_evaluation_metric = evaluation_metric

    # Just the default implementation.
    # def construct_critic_training(self):
    #     pass


        
        
class MultiVAEWithPhase4LambdaRank(MultiVAE):

    """After pre-training a MultiVAE, this allows you to fine-tune the results with the LambdaRank objective"""

    def superclass_stuff(self):
        self.create_second_actor_error()
        self.create_second_logging_ops()
        self.create_training_op_for_second_actor()

    def create_second_actor_error(self):
        self.second_actor_error_mask = tf.identity(self.batch_of_users)
        second_actor_error = my_lambdarank(self.prediction, self.second_actor_error_mask)
        second_actor_error = tf.reduce_sum(second_actor_error, axis=-1)
        mean_second_actor_error = tf.reduce_mean(second_actor_error)

        self.second_actor_error = second_actor_error
        self.mean_second_actor_error = mean_second_actor_error

    def create_training_op_for_second_actor(self):
        """As written now, must have same LR as original actor..."""
        train_op = tf.train.AdamOptimizer(self.lr_actor).minimize(
            self.mean_second_actor_error, var_list=self.actor_forward_variables)  #TODO: var_list part

        self.second_actor_train_op = train_op

    def create_second_logging_ops(self):
        tf.summary.scalar('mean_second_actor_error', self.mean_second_actor_error)  # Includes KL term...

        pass


class MultiVAEWithPhase4WARP(MultiVAE):

    """After pre-training a MultiVAE, this allows you to fine-tune the results with the WARP objective"""

    def superclass_stuff(self):
        self.create_second_actor_error()
        self.create_second_logging_ops()
        self.create_training_op_for_second_actor()

    def create_second_actor_error(self):
        self.second_actor_error_mask = tf.identity(self.batch_of_users)
        # self.error_vector_creator = ErrorVectorCreator(
        #     input_dim=self.input_dim, limit=self.error_vector_limit)
        self.second_error_vector_creator = ErrorVectorCreator(input_dim=self.input_dim)
        error_scaler = tf.py_func(self.second_error_vector_creator,
                                  [self.prediction, self.second_actor_error_mask], tf.float32)

        true_second_error = self.prediction * error_scaler

        self.second_actor_error = tf.reduce_sum(true_second_error, axis=-1)
        # self.mean_second_actor_error = tf.reduce_mean(
        #     self.second_actor_error) + (self.actor_reg_loss_scaler * self.actor_regularization_loss)
        self.mean_second_actor_error = tf.reduce_mean(self.second_actor_error)


    def create_training_op_for_second_actor(self):
        """As written now, must have same LR as original actor..."""
        train_op = tf.train.AdamOptimizer(self.lr_actor).minimize(
            self.mean_second_actor_error, var_list=self.actor_forward_variables)  #TODO: var_list part

        self.second_actor_train_op = train_op


    def create_second_logging_ops(self):
        tf.summary.scalar('mean_second_actor_error', self.mean_second_actor_error)  # Includes KL term...

        pass



class LambdaRankEncoder(CriticModelMixin, StochasticActorModelMixin, BaseModel):

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            anneal_cap=0.2,
            epochs_to_anneal_over=50,
            batch_size=500,
            evaluation_metric='NDCG',
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            #  actor_reg_loss_scaler=1e-4, #This is the KL scaler... And it varies as you go. So annoying.
            #  ac_reg_loss_scaler=1e-4,
            #  ac_reg_loss_scaler=0.2, #This uses the KL loss on the AC training.
            ac_reg_loss_scaler=0.0,  # We'll increase it as need be.
            omit_num_seen_from_critic=False,
            omit_num_unseen_from_critic=False,
            **kwargs):
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

        if hasattr(self, 'superclass_stuff'):
            self.superclass_stuff()

    def construct_actor_error(self):
        # NOTE: Check the KL term later.
        # NOTE: This will give the wrong number for validation stuff, because batch_of_users isn't the right output!.


        self.actor_error_mask = tf.identity(self.batch_of_users)
        actor_error = my_lambdarank(self.prediction, self.actor_error_mask)
        actor_error = tf.reduce_sum(actor_error, axis=-1)

        mean_actor_error = tf.reduce_mean(actor_error) + (self.kl_loss_scaler * self.actor_regularization_loss)

        self.actor_error = actor_error
        self.mean_actor_error = mean_actor_error


    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)


class WarpEncoder(CriticModelMixin, LinearModelMixin, BaseModel):

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            #  anneal_cap=0.2,
            #  epochs_to_anneal_over=50,
            # error_vector_limit=100,
            evaluation_metric='NDCG',
            batch_size=500,
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            # ac_reg_loss_scaler=1.0, # It's already scaled...
            ac_reg_loss_scaler=0.0,  #Just to be ...safe.
            actor_reg_loss_scaler=1e-4,
            **kwargs):
        """
        I'll do a better job about defining the inputs here.
        """
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        self.actor_error_mask = tf.identity(self.batch_of_users)
        # self.error_vector_creator = ErrorVectorCreator(
        #     input_dim=self.input_dim, limit=self.error_vector_limit)
        self.error_vector_creator = ErrorVectorCreator(input_dim=self.input_dim)
        error_scaler = tf.py_func(self.error_vector_creator,
                                  [self.prediction, self.actor_error_mask], tf.float32)

        true_error = self.prediction * error_scaler

        self.actor_error = tf.reduce_sum(true_error, axis=-1)
        print("Shape of actor_error should be like 500: {}".format(self.actor_error.get_shape()))
        self.mean_actor_error = tf.reduce_mean(
            self.actor_error) + (self.actor_reg_loss_scaler * self.actor_regularization_loss)

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)  #Includes regularization.


class WeightedMatrixFactorization(CriticModelMixin, LinearModelMixin, BaseModel):

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            #  anneal_cap=0.2,
            #  epochs_to_anneal_over=50,
            # error_vector_limit=100,
            positive_weights=2.0,
            evaluation_metric='NDCG',
            batch_size=500,
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            # ac_reg_loss_scaler=1.0, # It's already scaled...
            ac_reg_loss_scaler=0.0,  #Just to be ...safe.
            actor_reg_loss_scaler=1e-4,
            **kwargs):
        """
        Positive weights is how much bigger the positives are than the negatives.
        so, if it's 1, then the click-matrix will have 1 for negative, and 2 for positive.
        """
        local_variables = locals()
        assert positive_weights >= 2.0
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        self.actor_error_mask = tf.identity(self.batch_of_users)

        # the error mask is all 0 and 1s. So, I'll multiply it by positive_weights, and then add one.
        error_scaler = (self.actor_error_mask * (self.positive_weights - 1)) + 1

        square_difference = tf.square(self.actor_error_mask - self.prediction)
        true_error = error_scaler * square_difference

        self.actor_error = tf.reduce_sum(true_error, axis=-1)
        print("Shape of actor_error should be like 500: {}".format(self.actor_error.get_shape()))

        self.mean_actor_error = tf.reduce_mean(
            self.actor_error) + (self.actor_reg_loss_scaler * self.actor_regularization_loss)

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)  #Includes regularization.


class ProperlyShapedMultiDAE(CriticModelMixin, ProperlyShapedPointEstimateModelMixin, BaseModel):
    """This looks an awful lot like MultiDAE, except for the inheritances! It uses a more complicated base model."""

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            anneal_cap=0.2,
            epochs_to_anneal_over=50,
            batch_size=500,
            evaluation_metric='NDCG',
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            actor_reg_loss_scaler=1e-4,  #The default in training is 1e-4 so I'll leave it like that.
            #  ac_reg_loss_scaler=1e-4,
            #  ac_reg_loss_scaler=0.2, #This uses the KL loss on the AC training.
            ac_reg_loss_scaler=0.0,  # We'll increase it as need be.
            **kwargs):
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        # NOTE: Check the KL term later.
        # NOTE: This will give the wrong number for validation stuff, because batch_of_users isn't the right output!.

        self.actor_error_mask = tf.identity(self.batch_of_users)

        log_softmax = tf.nn.log_softmax(self.prediction)
        actor_error = -tf.reduce_sum(log_softmax * self.actor_error_mask, axis=-1)

        # This way, KL isn't factored into the critic at all. Which is probably what we want, although the AC should have it.
        # mean_actor_error = tf.reduce_mean(actor_error) + (
        #     self.kl_loss_scaler * self.actor_regularization_loss)
        mean_actor_error = tf.reduce_mean(actor_error) + (
            self.actor_reg_loss_scaler * self.actor_regularization_loss)

        self.actor_error = actor_error
        self.mean_actor_error = mean_actor_error

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)

        # tf.summary.scalar("NDCG@200", tf.reduce_mean(self.true_ndcg_at_200))
        # tf.summary.scalar("NDCG@50", tf.reduce_mean(self.true_ndcg_at_50))
        # tf.summary.scalar("NDCG@20", tf.reduce_mean(self.true_ndcg_at_20))
        # tf.summary.scalar("NDCG@5", tf.reduce_mean(self.true_ndcg_at_5))


class GaussianVAE(CriticModelMixin, StochasticActorModelMixin, BaseModel):

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            #  anneal_cap=0.2,
            #  epochs_to_anneal_over=50,
            # error_vector_limit=100,
            positive_weights=2.0,
            evaluation_metric='NDCG',
            batch_size=500,
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            # ac_reg_loss_scaler=1.0, # It's already scaled...
            ac_reg_loss_scaler=0.0,  #Just to be ...safe.
            actor_reg_loss_scaler=1e-4,
            **kwargs):
        """
        Positive weights is how much bigger the positives are than the negatives.
        so, if it's 1, then the click-matrix will have 1 for negative, and 2 for positive.
        """
        local_variables = locals()
        assert positive_weights >= 2.0
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        self.actor_error_mask = tf.identity(self.batch_of_users)

        # the error mask is all 0 and 1s. So, I'll multiply it by positive_weights, and then add one.
        error_scaler = (self.actor_error_mask * (self.positive_weights - 1)) + 1

        square_difference = tf.square(self.actor_error_mask - self.prediction)
        true_error = error_scaler * square_difference

        self.actor_error = tf.reduce_sum(true_error, axis=-1)
        print("Shape of actor_error should be like 500: {}".format(self.actor_error.get_shape()))

        self.mean_actor_error = tf.reduce_mean(
            self.actor_error) + (self.kl_loss_scaler * self.actor_regularization_loss)

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)  #Includes regularization.


class VWMF(CriticModelMixin, StochasticLinearActorModelMixin, BaseModel):

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            #  anneal_cap=0.2,
            #  epochs_to_anneal_over=50,
            # error_vector_limit=100,
            positive_weights=2.0,
            evaluation_metric='NDCG',
            batch_size=500,
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            # ac_reg_loss_scaler=1.0, # It's already scaled...
            ac_reg_loss_scaler=0.0,  #Just to be ...safe.
            actor_reg_loss_scaler=1e-4,
            **kwargs):
        """
        Positive weights is how much bigger the positives are than the negatives.
        so, if it's 1, then the click-matrix will have 1 for negative, and 2 for positive.
        """
        local_variables = locals()
        assert positive_weights >= 2.0
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        self.actor_error_mask = tf.identity(self.batch_of_users)

        # the error mask is all 0 and 1s. So, I'll multiply it by positive_weights, and then add one.
        error_scaler = (self.actor_error_mask * (self.positive_weights - 1)) + 1

        square_difference = tf.square(self.actor_error_mask - self.prediction)
        true_error = error_scaler * square_difference

        self.actor_error = tf.reduce_sum(true_error, axis=-1)
        print("Shape of actor_error should be like 500: {}".format(self.actor_error.get_shape()))

        self.mean_actor_error = tf.reduce_mean(
            self.actor_error) + (self.kl_loss_scaler * self.actor_regularization_loss)

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)  #Includes regularization.


class VariationalWarpEncoder(CriticModelMixin, StochasticActorModelMixin, BaseModel):

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            #  anneal_cap=0.2,
            #  epochs_to_anneal_over=50,
            # error_vector_limit=100,
            evaluation_metric='NDCG',
            batch_size=500,
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            # ac_reg_loss_scaler=1.0, # It's already scaled...
            ac_reg_loss_scaler=0.0,  #Just to be ...safe.
            actor_reg_loss_scaler=1e-4,
            **kwargs):
        """
        I'll do a better job about defining the inputs here.
        """
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        self.actor_error_mask = tf.identity(self.batch_of_users)
        # self.error_vector_creator = ErrorVectorCreator(
        #     input_dim=self.input_dim, limit=self.error_vector_limit)
        self.error_vector_creator = ErrorVectorCreator(input_dim=self.input_dim)
        error_scaler = tf.py_func(self.error_vector_creator,
                                  [self.prediction, self.actor_error_mask], tf.float32)

        true_error = self.prediction * error_scaler

        self.actor_error = tf.reduce_sum(true_error, axis=-1)
        print("Shape of actor_error should be like 500: {}".format(self.actor_error.get_shape()))
        self.mean_actor_error = tf.reduce_mean(
            self.actor_error) + (self.kl_loss_scaler * self.actor_regularization_loss)

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)  #Includes regularization.
