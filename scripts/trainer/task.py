

# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import time

from absl import app
from absl import flags
from absl import logging
from official.nlp import optimization 

import random
import string

from google.cloud import aiplatform as vertex_ai


TFHUB_HANDLE_ENCODER = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
TFHUB_HANDLE_PREPROCESS = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
LOCAL_TB_FOLDER = '/tmp/logs'
LOCAL_SAVED_MODEL_DIR = '/tmp/saved_model'

FLAGS = flags.FLAGS
flags.DEFINE_integer('steps_per_epoch', 625, 'Steps per training epoch')
flags.DEFINE_integer('eval_steps', 150, 'Evaluation steps')
flags.DEFINE_integer('epochs', 2, 'Nubmer of epochs')
flags.DEFINE_integer('per_replica_batch_size', 32, 'Per replica batch size')
flags.DEFINE_integer('TRAIN_NGPU', 1, '')
flags.DEFINE_integer('replica_count', 1, '')
flags.DEFINE_integer('reduction_cnt', 0, '')

flags.DEFINE_string('training_data_path', f'/bert-finetuning/imdb/tfrecords/train', 'Training data GCS path')
flags.DEFINE_string('validation_data_path', f'/bert-finetuning/imdb/tfrecords/valid', 'Validation data GCS path')
flags.DEFINE_string('testing_data_path', f'/bert-finetuning/imdb/tfrecords/test', 'Testing data GCS path')

flags.DEFINE_string('job_dir', f'/jobs', 'A base GCS path for jobs')
flags.DEFINE_string('job_id', 'default', 'unique_id for experiment runs')
flags.DEFINE_string('TRAIN_GPU', 'NA', '')
flags.DEFINE_string('experiment_run', 'NA', '')
flags.DEFINE_string('experiment_name', 'NA', '')


flags.DEFINE_enum('strategy', 'multiworker', ['single', 'mirrored', 'multiworker'], 'Distribution strategy')
flags.DEFINE_enum('auto_shard_policy', 'auto', ['auto', 'data', 'file', 'off'], 'Dataset sharing strategy')

auto_shard_policy = {
    'auto': tf.data.experimental.AutoShardPolicy.AUTO,
    'data': tf.data.experimental.AutoShardPolicy.DATA,
    'file': tf.data.experimental.AutoShardPolicy.FILE,
    'off': tf.data.experimental.AutoShardPolicy.OFF,
}


def create_unbatched_dataset(tfrecords_folder):
    """Creates an unbatched dataset in the format required by the 
       sentiment analysis model from the folder with TFrecords files."""
    
    feature_description = {
        'text_fragment': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    def _parse_function(example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        return parsed_example['text_fragment'], parsed_example['label']
  
    file_paths = [f'{tfrecords_folder}/{file_path}' for file_path in tf.io.gfile.listdir(tfrecords_folder)]
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_parse_function)
    
    return dataset


def configure_dataset(ds, auto_shard_policy):
    """
    Optimizes the performance of a dataset.
    """
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        auto_shard_policy
    )
    
    ds = ds.repeat(-1).cache()
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    ds = ds.with_options(options)
    return ds


def create_input_pipelines(train_dir, valid_dir, test_dir, batch_size, auto_shard_policy):
    """Creates input pipelines from Imdb dataset."""
    
    train_ds = create_unbatched_dataset(train_dir)
    train_ds = train_ds.batch(batch_size)
    train_ds = configure_dataset(train_ds, auto_shard_policy)
    
    valid_ds = create_unbatched_dataset(valid_dir)
    valid_ds = valid_ds.batch(batch_size)
    valid_ds = configure_dataset(valid_ds, auto_shard_policy)
    
    test_ds = create_unbatched_dataset(test_dir)
    test_ds = test_ds.batch(batch_size)
    test_ds = configure_dataset(test_ds, auto_shard_policy)

    return train_ds, valid_ds, test_ds


def build_classifier_model(tfhub_handle_preprocess, tfhub_handle_encoder):
    """Builds a simple binary classification model with BERT trunk."""
    
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    
    return tf.keras.Model(text_input, net)


def copy_tensorboard_logs(local_path: str, gcs_path: str):
    """Copies Tensorboard logs from a local dir to a GCS location.
    
    After training, batch copy Tensorboard logs locally to a GCS location. This can result
    in faster pipeline runtimes over streaming logs per batch to GCS that can get bottlenecked
    when streaming large volumes.
    
    Args:
      local_path: local filesystem directory uri.
      gcs_path: cloud filesystem directory uri.
    Returns:
      None.
    """
    pattern = '{}/*/events.out.tfevents.*'.format(local_path)
    local_files = tf.io.gfile.glob(pattern)
    gcs_log_files = [local_file.replace(local_path, gcs_path) for local_file in local_files]
    for local_file, gcs_file in zip(local_files, gcs_log_files):
        tf.io.gfile.copy(local_file, gcs_file)


def main(argv):
    del argv
    
    def _is_chief(task_type, task_id):
        return ((task_type == 'chief' or task_type == 'worker') and task_id == 0) or task_type is None
        
    
    logging.info('Setting up training.')
    logging.info('   epochs: {}'.format(FLAGS.epochs))
    logging.info('   steps_per_epoch: {}'.format(FLAGS.steps_per_epoch))
    logging.info('   eval_steps: {}'.format(FLAGS.eval_steps))
    logging.info('   strategy: {}'.format(FLAGS.strategy))
    logging.info('   job_id: {}'.format(FLAGS.job_id))
    logging.info('   TRAIN_GPU: {}'.format(FLAGS.TRAIN_GPU))
    logging.info('   TRAIN_NGPU: {}'.format(FLAGS.TRAIN_NGPU))
    logging.info('   replica_count: {}'.format(FLAGS.replica_count))
    logging.info('   reduction_cnt: {}'.format(FLAGS.reduction_cnt))
    logging.info('   experiment_name: {}'.format(FLAGS.experiment_name))
    logging.info('   experiment_run: {}'.format(FLAGS.experiment_run))
    
    tb_dir = os.getenv('AIP_TENSORBOARD_LOG_DIR', LOCAL_TB_FOLDER)
    model_dir = os.getenv('AIP_MODEL_DIR', LOCAL_SAVED_MODEL_DIR)
    logging.info(f'AIP_TENSORBOARD_LOG_DIR = {tb_dir}')
    logging.info(f'AIP_MODEL_DIR = {model_dir}')
    
    project_number = os.environ["CLOUD_ML_PROJECT_ID"]
    
    vertex_ai.init(
        project=project_number,
        location='us-central1',
        experiment=FLAGS.experiment_name
    )

    # Single Machine, single compute device
    if FLAGS.strategy == 'single':
        if tf.test.is_gpu_available():
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        logging.info("Single device training")
    
    # Single Machine, multiple compute device
    elif FLAGS.strategy == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
        logging.info("Mirrored Strategy distributed training")
    
    # Multi Machine, multiple compute device
    elif FLAGS.strategy == 'multiworker':
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        logging.info("Multi-worker Strategy distributed training")
        logging.info('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
   
    # Single Machine, multiple TPU devices
    elif FLAGS.strategy == 'tpu':
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))

    logging.info('num_replicas_in_sync = {}'.format(strategy.num_replicas_in_sync))
    
    if strategy.cluster_resolver:    
        task_type, task_id = (strategy.cluster_resolver.task_type,
                              strategy.cluster_resolver.task_id)
    else:
        task_type, task_id =(None, None)
        
    logging.info('task_type = {}'.format(task_type))
    logging.info('task_id = {}'.format(task_id))
    
    global_batch_size = (
        strategy.num_replicas_in_sync *
        FLAGS.per_replica_batch_size
    )
    
    train_ds, valid_ds, test_ds = create_input_pipelines(
        FLAGS.training_data_path,
        FLAGS.validation_data_path,
        FLAGS.testing_data_path,
        global_batch_size,
        auto_shard_policy[FLAGS.auto_shard_policy])
        
    num_train_steps = FLAGS.steps_per_epoch * FLAGS.epochs
    num_warmup_steps = int(0.1*num_train_steps)
    init_lr = 3e-5
    
    with strategy.scope():
        
        model = build_classifier_model(TFHUB_HANDLE_PREPROCESS, TFHUB_HANDLE_ENCODER)
        
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        metrics = tf.metrics.BinaryAccuracy()
        
        optimizer = optimization.create_optimizer(
            init_lr=init_lr
            , num_train_steps=num_train_steps
            , num_warmup_steps=num_warmup_steps
            , optimizer_type='adamw'
        )

        model.compile(
            optimizer=optimizer
            , loss=loss
            , metrics=metrics
        )
        
    # Configure BackupAndRestore callback
    if FLAGS.strategy == 'single':
        callbacks = []
        logging.info("No backup and restore")
    else:
        backup_dir = '{}/backupandrestore'.format(FLAGS.job_dir)
        callbacks = [tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir)]
        logging.info(f"saved backup and restore t0: {backup_dir}")
    
    # Configure TensorBoard callback on Chief
    if _is_chief(task_type, task_id):
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=tb_dir
                , update_freq='batch'
                , histogram_freq=1
            )
        )
    
    logging.info('Starting training ...')
    
    if _is_chief(task_type, task_id):
        start_time = time.time()
    
    history = model.fit(
        x=train_ds
        , validation_data=valid_ds
        , steps_per_epoch=FLAGS.steps_per_epoch
        , validation_steps=FLAGS.eval_steps
        , epochs=FLAGS.epochs
        , callbacks=callbacks
    )
    
    # ====================================================
    # log Vertex Experiments
    # ====================================================
    SESSION_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=3))
    
    if _is_chief(task_type, task_id):
        end_time = time.time()
        # val metrics
        val_keys = [v for v in history.history.keys()]
        total_train_time = int((end_time - start_time) / 60)

        metrics_dict = {"total_train_time": total_train_time}
        logging.info(f"total_train_time: {total_train_time}")
        _ = [metrics_dict.update({key: history.history[key][-1]}) for key in val_keys]
    
        logging.info(f" task_type logging experiments: {task_type}")
        logging.info(f" task_id logging experiments: {task_id}")
        logging.info(f" logging data to experiment run: {FLAGS.experiment_run}-{SESSION_id}")
        
        with vertex_ai.start_run(
            f'{FLAGS.experiment_run}-{SESSION_id}', 
        ) as my_run:
            
            logging.info(f"logging metrics...")
            my_run.log_metrics(metrics_dict)

            logging.info(f"logging metaparams...")
            my_run.log_params(
                {
                    "epochs": FLAGS.epochs,
                    "strategy": FLAGS.strategy,
                    "per_replica_batch_size": FLAGS.per_replica_batch_size,
                    "TRAIN_GPU": FLAGS.TRAIN_GPU,
                    "TRAIN_NGPU": FLAGS.TRAIN_NGPU,
                    "replica_count": FLAGS.replica_count,
                    "reduction_cnt": FLAGS.reduction_cnt,
                    "global_batch_size": global_batch_size,
                }
            )

            vertex_ai.end_run()
            logging.info(f"EXPERIMENT RUN: '{FLAGS.experiment_run}-{SESSION_id}' has ended")

    if _is_chief(task_type, task_id):
        # Copy tensorboard logs to GCS
        # tb_logs = '{}/tb_logs'.format(FLAGS.job_dir)
        # logging.info('Copying TensorBoard logs to: {}'.format(tb_logs))
        # copy_tensorboard_logs(LOCAL_TB_FOLDER, tb_logs)
        # saved_model_dir = '{}/saved_model'.format(model_dir)
        logging.info('Training completed. Saving the trained model to: {}'.format(model_dir))
        model.save(model_dir)
    # else:
    #     # saved_model_dir = model_dir
    #     logging.info('Training completed. Saving the trained model to: {}'.format(model_dir))
    #     model.save(model_dir)
        
    # Save trained model
    # saved_model_dir = '{}/saved_model'.format(model_dir)
    # logging.info('Training completed. Saving the trained model to: {}'.format(saved_model_dir))
    # model.save(saved_model_dir)
    #tf.saved_model.save(model, saved_model_dir)
    
    
if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
