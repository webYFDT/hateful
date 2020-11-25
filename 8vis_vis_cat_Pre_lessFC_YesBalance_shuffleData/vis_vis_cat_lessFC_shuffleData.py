# Copyright (c) Facebook, Inc. and its affiliates.

import warnings
import random

import os
from copy import deepcopy

import torch
from omegaconf import OmegaConf
from torch import nn
from transformers.modeling_bert import (
    BertConfig,
    BertEncoder,
    BertForPreTraining,
    BertLayer,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)

from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.embeddings import BertVisioLinguisticEmbeddings
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)


class VisualBERTBase(BertPreTrainedModel):
    def __init__(
        self,
        config,
        visual_embedding_dim=512,
        embedding_strategy="plain",
        bypass_transformer=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        super().__init__(config)
        self.config = config

        config.visual_embedding_dim = visual_embedding_dim
        config.embedding_strategy = embedding_strategy
        config.bypass_transformer = bypass_transformer
        config.output_attentions = output_attentions
        config.output_hidden_states = output_hidden_states

        self.embeddings = BertVisioLinguisticEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.bypass_transformer = config.bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = BertLayer(config)

        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.fixed_head_masks = [None for _ in range(len(self.encoder.layer))]
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids,
            token_type_ids,
            visual_embeddings=visual_embeddings,
            position_embeddings_visual=position_embeddings_visual,
            visual_embeddings_type=visual_embeddings_type,
            image_text_alignment=image_text_alignment,
        )

        if self.bypass_transformer and visual_embeddings is not None:
            assert (
                not self.output_hidden_states
            )  # Don't support this for the bypass model
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_part = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[
                :, :, :text_length, :text_length
            ]

            encoded_layers = self.encoder(
                text_embedding_output,
                text_extended_attention_mask,
                self.fixed_head_masks,
            )
            sequence_output = encoded_layers[0]
            new_input = torch.cat((sequence_output, visual_part), dim=1)
            final_sequence_output = self.additional_layer(
                new_input, extended_attention_mask
            )
            pooled_output = self.pooler(final_sequence_output)
            return final_sequence_output, pooled_output

        if self.output_attentions:
            encoded_layers = self.encoder(
                embedding_output, extended_attention_mask, self.fixed_head_masks
            )
            sequence_output = encoded_layers[0]
            attn_data_list = encoded_layers[1:]
            pooled_output = self.pooler(sequence_output)
            return encoded_layers, pooled_output, attn_data_list
        else:
            encoded_layers = self.encoder(
                embedding_output, extended_attention_mask, self.fixed_head_masks
            )
            sequence_output = encoded_layers[0]
            pooled_output = self.pooler(sequence_output)
            return sequence_output, pooled_output, []


class VisualBERTForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states

        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = getattr(self.config, "bert_model_name", None)
        self.bert_config = BertConfig.from_dict(
            OmegaConf.to_container(self.config, resolve=True)
        )
        if self.bert_model_name is None:
            self.bert = VisualBERTBase(
                self.bert_config,
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )
        else:
            self.bert = VisualBERTBase.from_pretrained(
                self.config.bert_model_name,
                config=self.bert_config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )

        self.vocab_size = self.bert.config.vocab_size

        # TODO: Once omegaconf fixes int keys issue, bring this back
        # See https://github.com/omry/omegaconf/issues/149
        # with omegaconf.open_dict(self.config):
        #     # Add bert config such as hidden_state to our main config
        #     self.config.update(self.bert.config.to_dict())
        if self.bert_model_name is None:
            bert_masked_lm = BertForPreTraining(self.bert.config)
        else:
            bert_masked_lm = BertForPreTraining.from_pretrained(
                self.config.bert_model_name,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
            )
        self.cls = deepcopy(bert_masked_lm.cls)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()
                self.cls.apply(self.bert._init_weights)

            self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them
            instead.
        """
        self.bert._tie_or_clone_weights(
            self.cls.predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def forward(
        self,
        input_ids,
        input_mask,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
        masked_lm_labels=None,
    ):
        sequence_output, pooled_output, attention_weights = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            position_embeddings_visual,
            visual_embeddings_type,
            image_text_alignment,
        )

        output_dict = {}

        if self.output_attentions:
            output_dict["attention_weights"] = attention_weights

        if self.output_hidden_states:
            output_dict["sequence_output"] = sequence_output
            output_dict["pooled_output"] = pooled_output

        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )
        if masked_lm_labels is not None:
            output_dict["logits"] = prediction_scores
            masked_lm_loss = self.loss_fct(
                prediction_scores.contiguous().view(-1, self.vocab_size),
                masked_lm_labels.contiguous().view(-1),
            )
            output_dict["masked_lm_loss"] = masked_lm_loss
            output_dict["loss"] = masked_lm_loss

        return output_dict


class VisualBERTForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.pooler_strategy = self.config.get("pooler_strategy", "default")

        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = getattr(self.config, "bert_model_name", None)
        self.bert_config = BertConfig.from_dict(
            OmegaConf.to_container(self.config, resolve=True)
        )
        if self.bert_model_name is None:
            self.bert = VisualBERTBase(
                self.bert_config,
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )
        else:
            self.bert = VisualBERTBase.from_pretrained(
                self.config.bert_model_name,
                config=self.bert_config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )

        self.training_head_type = self.config.training_head_type
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        if self.config.training_head_type == "nlvr2":
            self.bert.config.hidden_size *= 2
        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.bert.config))
        '''
            nn.Linear(self.bert.config.hidden_size, self.config.num_labels)
        )
        '''
        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()

            # Classifier needs to be initialized always as it is task specific
            self.classifier.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids,
        input_mask,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
        masked_lm_labels=None,
    ):
        sequence_output, pooled_output, attention_weights = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            position_embeddings_visual,
            visual_embeddings_type,
            image_text_alignment,
        )

        if self.training_head_type == "nlvr2":
            # 2B * H => B * 2H
            b, h = pooled_output.size()
            pooled_output = torch.cat(
                [pooled_output[: b // 2], pooled_output[b // 2 :]], dim=1
            )

        output_dict = {}
        if self.output_attentions:
            output_dict["attention_weights"] = attention_weights

        if self.output_hidden_states:
            output_dict["sequence_output"] = sequence_output
            output_dict["pooled_output"] = pooled_output

        if self.pooler_strategy == "vqa":
            # In VQA2 pooling strategy, we use representation from second last token
            index_to_gather = input_mask.sum(1) - 2
            pooled_output = torch.gather(
                sequence_output,
                1,
                index_to_gather.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(index_to_gather.size(0), 1, sequence_output.size(-1)),
            )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        '''
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output_dict["scores"] = reshaped_logits
        return output_dict
        '''
        return logits


#@registry.register_model("visual_bert")
class VisualBERT(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.build()
        
    @classmethod
    def config_path(cls):
        return "configs/models/visual_bert/pretrain.yaml"

    def build(self):
        if self.config.training_head_type == "pretraining":
            self.model = VisualBERTForPretraining(self.config)
        else:
            self.model = VisualBERTForClassification(self.config)
            #warnings.warn('vis*'*100)
            #warnings.warn(str(self.model))
            pre_vis=torch.load('/root/.cache/torch/mmf/data/models/visual_bert.finetuned.hateful_memes.from_coco/model.pth')
            '''
            model_vis=self.model.state_dict()
            warnings.warn("pre_key"*100)
            warnings.warn(str(pre_vis.keys()))
            warnings.warn("vis_key"*100)  
            warnings.warn(str(model_vis.keys()))
            '''
            model_vis=self.model.state_dict()
            #warnings.warn("pre_key"*100)
            #warnings.warn(str(pre_vil.keys()))
            #warnings.warn("vil_key"*100)
            #warnings.warn(str(model_vil.keys()))            
            pre_train_dict={k[6:]: v for k, v in pre_vis.items() if k[6:] in model_vis}
            '''
            warnings.warn("vis_pre_train_dict")
            warnings.warn(str(pre_train_dict))
            '''
            model_vis.update(pre_train_dict)
            self.model.load_state_dict(model_vis)
            
        if self.config.special_visual_initialize:
            self.model.bert.embeddings.initialize_visual_from_pretrained()

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def flatten(self, sample_list, to_be_flattened=None, to_be_flattened_dim=None):
        if to_be_flattened is None:
            to_be_flattened = {}
        if to_be_flattened_dim is None:
            to_be_flattened_dim = {}
        for key in to_be_flattened:
            # Make sure these keys are present or otherwise set these keys to None
            sample_list[key] = getattr(sample_list, key, None)
            sample_list[key] = transform_to_batch_sequence(sample_list[key])
        for key in to_be_flattened_dim:
            sample_list[key] = getattr(sample_list, key, None)
            sample_list[key] = transform_to_batch_sequence_dim(sample_list[key])

        if sample_list.visual_embeddings_type is None:
            if sample_list.image_mask is not None:
                sample_list.visual_embeddings_type = torch.zeros_like(
                    sample_list.image_mask, dtype=torch.long
                )

        if sample_list.image_mask is not None:
            attention_mask = torch.cat(
                (sample_list.input_mask, sample_list.image_mask), dim=-1
            )
            if sample_list.masked_lm_labels is not None:
                assert sample_list.masked_lm_labels.size(
                    -1
                ) == sample_list.input_mask.size(-1)
                new_lm_labels = torch.ones_like(attention_mask) * -1
                size_masked_lm_labels = sample_list.masked_lm_labels.size()
                assert len(size_masked_lm_labels) == 2
                new_lm_labels[
                    : size_masked_lm_labels[0], : size_masked_lm_labels[1]
                ] = sample_list.masked_lm_labels
                sample_list.masked_lm_labels = new_lm_labels
        else:
            attention_mask = sample_list.input_mask

        sample_list.attention_mask = attention_mask

        return sample_list

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def flatten_for_bert(self, sample_list, **kwargs):
        to_be_flattened = [
            "input_ids",
            "token_type_ids",
            "input_mask",
            "image_mask",
            "masked_lm_labels",
            "position_embeddings_visual",
            "visual_embeddings_type",
        ]
        to_be_flattened_dim = ["image_text_alignment", "visual_embeddings"]

        # We want to convert everything into: batch x sequence_length x (dim).
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def update_sample_list_based_on_head(self, sample_list):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids

        if self.config.training_head_type == "nlvr2":
            bert_input_ids = torch.cat([bert_input_ids, bert_input_ids])
            bert_input_mask = torch.cat([bert_input_mask, bert_input_mask])
            bert_input_type_ids = torch.cat([bert_input_type_ids, bert_input_type_ids])

            # image input
            img0 = getattr(sample_list, "img0", {})
            image_info = getattr(img0, "image_info_0", {})
            image_dim_variable_0 = getattr(image_info, "max_features", None)
            image_feat_variable_0 = getattr(img0, "image_feature_0", None)

            img1 = getattr(sample_list, "img1", {})
            image_info = getattr(img1, "image_info_0", {})
            image_dim_variable_1 = getattr(image_info, "max_features", None)
            image_feat_variable_1 = getattr(img1, "image_feature_0", None)

            image_feat_variable = torch.cat(
                [image_feat_variable_0, image_feat_variable_1]
            )
            image_dim_variable = torch.cat([image_dim_variable_0, image_dim_variable_1])
        else:
            image_info = getattr(sample_list, "image_info_0", {})
            image_dim_variable = getattr(image_info, "max_features", None)
            image_feat_variable = getattr(sample_list, "image_feature_0", None)

        sample_list.visual_embeddings = image_feat_variable
        sample_list.image_dim = image_dim_variable
        sample_list.input_ids = bert_input_ids
        sample_list.input_mask = bert_input_mask
        sample_list.token_type_ids = bert_input_type_ids
        return sample_list

    def add_custom_params(self, sample_list):
        visual_embeddings = getattr(sample_list, "visual_embeddings", None)
        image_dim = getattr(sample_list, "image_dim", None)
        # pretraining labels
        sample_list.masked_lm_labels = getattr(sample_list, "lm_label_ids", None)
        # image_feat_variable = batch x ( num_choice x ) image_feature_length x dim
        # Prepare Mask
        if visual_embeddings is not None and image_dim is not None:
            image_mask = (
                torch.arange(visual_embeddings.size(-2))
                .expand(*visual_embeddings.size()[:-1])
                .cuda()
            )
            if len(image_dim.size()) < len(image_mask.size()):
                image_dim = image_dim.unsqueeze(-1)
                assert len(image_dim.size()) == len(image_mask.size())
            image_mask = image_mask < image_dim
            sample_list.image_mask = image_mask.long()
        else:
            sample_list.image_mask = None

        sample_list.position_embeddings_visual = None

        return sample_list

    # Backward compatibility for code from original VisualBERT
    @classmethod
    def format_state_key(cls, key):
        return (
            key.replace("bert.bert", "model.bert")
            .replace("bert.cls", "model.cls")
            .replace("bert.classifier", "model.classifier")
        )
    
    def shuffle_sample_list(self,sample_list):
        '''
        warnings.warn('Vis_sample_list**********'*100)
        warnings.warn(str(sample_list.input_ids.shape)) 
        warnings.warn(str(sample_list.input_mask.shape))
        warnings.warn(str(sample_list.attention_mask.shape))
        warnings.warn(str(sample_list.token_type_ids.shape))
        warnings.warn(str(sample_list.visual_embeddings.shape))
        warnings.warn(str(sample_list.visual_embeddings_type.shape))
        warnings.warn(str(sample_list.masked_lm_labels.shape))
        '''
        '''
        warnings.warn(str(sample_list.visual_embeddings.shape))
        warnings.warn(str(sample_list.visual_embeddings))
        warnings.warn(str(sample_list.visual_embeddings_type))
        warnings.warn(str(sample_list.targets))
        '''
        
        
        index1=(sample_list.targets==1).nonzero()
        index0=(sample_list.targets==0).nonzero()
        
        index0_list=list(range(len(index0)))
        index1_list=list(range(len(index1)))
        
        random.shuffle(index0_list)
        random.shuffle(index1_list)
        
        
        index01_list=[]
        for i in sample_list.targets:
            if i.item()==1:
                num=index1_list.pop()
                num_i=index1[num].item()
            else :
                num=index0_list.pop()
                num_i=index0[num].item()
            index01_list.append(num_i)
        
        sample_list.input_ids=sample_list.input_ids[index01_list]
        sample_list.input_mask=sample_list.input_mask[index01_list]
        sample_list.attention_mask=sample_list.attention_mask[index01_list]
        sample_list.token_type_ids=sample_list.token_type_ids[index01_list]
        sample_list.visual_embeddings=sample_list.visual_embeddings[index01_list]
        sample_list.visual_embeddings_type=sample_list.visual_embeddings_type[index01_list]
        sample_list.masked_lm_labels=sample_list.masked_lm_labels[index01_list]
        sample_list.targets=sample_list.targets[index01_list]
        
        '''
        warnings.warn(str(sample_list.visual_embeddings.shape))
        warnings.warn(str(sample_list.visual_embeddings))        
        warnings.warn(str(sample_list.visual_embeddings_type))
        warnings.warn(str(sample_list.targets))
        
        '''
        '''
        warnings.warn(str(sample_list.position_embeddings_visual))
        warnings.warn(str(sample_list.image_text_alignment))
        os._exit()
        '''
        
        
        
        return sample_list
    
    def forward(self, sample_list):

        
        sample_list = self.update_sample_list_based_on_head(sample_list)
        sample_list = self.add_custom_params(sample_list)
        sample_list = self.flatten_for_bert(sample_list)
        
        
        #打乱数据集
        sample_list=self.shuffle_sample_list(sample_list)
        

        
        
        output_dict = self.model(
            sample_list.input_ids,
            sample_list.input_mask,
            sample_list.attention_mask,
            sample_list.token_type_ids,
            sample_list.visual_embeddings,
            sample_list.position_embeddings_visual, #
            sample_list.visual_embeddings_type,
            sample_list.image_text_alignment,      #
            sample_list.masked_lm_labels,
        )
        '''
        if "pretraining" in self.config.training_head_type:
            loss_key = "{}/{}".format(
                sample_list.dataset_name, sample_list.dataset_type
            )
            output_dict["losses"] = {}
            output_dict["losses"][loss_key + "/masked_lm_loss"] = output_dict.pop(
                "masked_lm_loss"
            )
        '''
        return output_dict




from mmf.utils.build import build_classifier_layer
@registry.register_model("vis_vis_cat_lessFC_shuffleData")
class VisVisCatLessFCshuffleData(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        warnings.warn('load_vis_vis_pretrain_LessFC_shuffle_data--------------'*100)
        self.build()
        
    @classmethod
    def config_path(cls):
        return "configs/models/vis_vis_cat_Pre_lessFC_noBalance_shuffleData/visvisc_model_config_lessFC_shuffleData.yaml"   
    def build(self):
        self.vis1=VisualBERT(self.config)
        self.vis2=VisualBERT(self.config)
        self.classifier = build_classifier_layer(self.config.classifier)
    def forward(self, sample_list):
        
        vis1=self.vis1.forward(sample_list)
        vis2=self.vis2.forward(sample_list)
        #warnings.warn('vis shape:'+str(vis.shape))
        #warnings.warn('vil shape:'+str(vil.shape))
        combined=torch.cat([vis1, vis2], dim=1)
        logits = self.classifier(combined)
        output = {"scores": logits}
        return output