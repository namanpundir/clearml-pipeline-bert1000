# -*- coding: utf-8 -*-
from clearml import PipelineDecorator
from mimetypes import init
from clearml.automation.controller import PipelineDecorator
from clearml import Task, TaskTypes, Logger
import argparse
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import transforms
from dataclasses import dataclass
#pip install transformers
# Variables for Pipeline Decorators

project = "Training_Pipeline"
pipeline = "Bert1000"
version = "0.0.1"




############################
# @PipelineDecorator.component(return_values=['final_x'], cache=True, task_type=TaskTypes.training)
@PipelineDecorator.component(return_values=['final_x'], cache=True, task_type=TaskTypes.training, execution_queue="training")

def train():
    import torch
    from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
    from transformers import BertTokenizerFast, BertForSequenceClassification
    from transformers import Trainer, TrainingArguments
    import numpy as np
    import random
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from torchvision import transforms
    from dataclasses import dataclass

    def set_seed(seed: int):
      random.seed(seed)
      np.random.seed(seed)
      if is_torch_available():
          torch.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
          # ^^ safe to call this function even if cuda is not available
      if is_tf_available():
          import tensorflow as tf

          tf.random.set_seed(seed)

    set_seed(1)



    import torch
    torch.cuda.empty_cache()
    from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
    from transformers import BertTokenizerFast, BertForSequenceClassification
    from transformers import Trainer, TrainingArguments
    import numpy as np
    import random
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from torchvision import transforms


    model_name = "bert-base-uncased"

    ####bert config just for reference
    '''
    ( vocab_size = 30522hidden_size = 768num_hidden_layers = 12num_attention_heads = 12
    intermediate_size = 3072hidden_act = 'gelu'hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1max_position_embeddings = 512
    type_vocab_size = 2initializer_range = 0.02layer_norm_eps = 1e-12pad_token_id = 0
    position_embedding_type = 'absolute'use_cache = Trueclassifier_dropout = None**kwargs )
    '''

    max_length = 512

    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

    test_size=0.2

    dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
    documents = dataset.data
    labels = dataset.target
    (train_texts, valid_texts, train_labels, valid_labels), target_names = train_test_split(documents, labels, test_size=test_size), dataset.target_names
    x_train, y_train, x_val, y_val = train_texts, valid_texts, train_labels, valid_labels

    train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(y_train, truncation=True, padding=True, max_length=max_length)


    ######################DATALOADER########################
    class NewsGroupsDataset(torch.utils.data.Dataset):
      def __init__(self, encodings, labels):
          self.encodings = encodings
          self.labels = labels

      def __getitem__(self, idx):
          item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
          item["labels"] = torch.tensor([self.labels[idx]])
          return item

      def __len__(self):
          return len(self.labels)

    # convert our tokenized data into a torch Dataset
    train_dataset = NewsGroupsDataset(train_encodings, train_labels)
    valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)
    #gpu
    # model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names)).to("cuda")

    #cpu
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names)).to("cpu")
    from sklearn.metrics import accuracy_score


    def compute_metrics(pred):
      labels = pred.label_ids
      preds = pred.predictions.argmax(-1)
      #     calculate accuracy using sklearn's function
      acc = accuracy_score(labels, preds)
      return {
          'accuracy': acc,
      }


    def train_model(train_dataset, valid_dataset):

      ##################################Train ARGUMENTS###############################
      training_args = TrainingArguments(
      output_dir='./results',

      ################change number of epochs for testing to 3 , for actual training 1000
      num_train_epochs=1000,
      per_device_train_batch_size=12,
      per_device_eval_batch_size=12,
      warmup_steps=500,
      weight_decay=0.01,
      learning_rate=0.0001,
      logging_dir='./logs',
      load_best_model_at_end=True,

      logging_steps=400,
      save_steps=400,
      evaluation_strategy="steps",
      )
      #################################################################################

      trainer = Trainer(
          model=model,
          args=training_args,
          train_dataset=train_dataset,
          eval_dataset=valid_dataset,
          compute_metrics=compute_metrics,
      )

      return trainer

        ####Train
    print("check1")
    train_model(train_dataset, valid_dataset).train()

    train_model(train_dataset, valid_dataset).evaluate()

      #####bin or pt file - used to serve in bittensor serving
    model_path = "bert_train"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    final_x = torch.save(model.state_dict(), 'bert-model.bin') #can remove if throws error

    return final_x


#############################################END OF TRAINING CODE###########################################################################

# Pipeline execution context
@PipelineDecorator.pipeline(name=pipeline, project=project, version=version, add_pipeline_tags=True, pipeline_execution_queue="training")
def executing_pipeline():
    # print('launch step one')
    # SEED = set_seed(self_seed)

    print('Train Step Launching')
    final_x = train()


if __name__ == '__main__':
  # set the pipeline steps default execution queue (per specific step we can override it with the decorator)
  #PipelineDecorator.set_default_execution_queue('training')
  # Run the pipeline steps as subprocesses on the current machine, great for local executions
  # (for easy development / debugging, use `PipelineDecorator.debug_pipeline()` to execute steps as regular functions)

  #
  executing_pipeline()

  print('process completed')

  #description = 'Train Example Pipeline'
  # defining command line parameters
  #parser = argparse.ArgumentParser(description=description)
  # parser.add_argument('--steps', type=int, default=200, help='steps per epoch for training (default: 200)')
  # parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
  # parser.add_argument('--project', type=str, default="keras_example", help='name of the project (default: keras_example)')
  # parser.add_argument('--pipeline', type=str, default="Keras Example Pipeline", help='name of the pipeline (default: Keras Example Pipeline)')
  # parser.add_argument('--version', type=str, default="0.0.1", help='version of the project (default: 0.0.1)')

  # parsing the arguments
  #args = parser.parse_args()

  #BittensorTraining(args=args).executing_pipeline()
