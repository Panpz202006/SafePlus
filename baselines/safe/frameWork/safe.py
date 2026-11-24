"""Train and evaluate the legacy SAFE fraud early-detection baseline."""
from collections import defaultdict
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

sys.path.append("../")
from safdKit import early_det, minibatch, prec_reca_F1, upp_tri_mat

# X contains per-account feature histories. T gives the observed length,
# and C is 1 for an observed event and 0 for a right-censored account.
if len(sys.argv) != 2:
    print "please add 1 parameter."
    exit(0)

if sys.argv[1] == "twitter":

    source_path = "../twitter/"
    X_train = np.load(source_path + "X_train_var.npy")
    T_train = np.load(source_path + "T_train_var.npy")
    C_train = np.load(source_path + "C_train_var.npy")

    X_test = np.load(source_path + "X_test_var.npy")
    T_test = np.load(source_path + "T_test_var.npy")
    C_test = np.load(source_path + "C_test_var.npy")

    X_valid = np.load(source_path + "X_valid_var.npy")
    T_valid = np.load(source_path + "T_valid_var.npy")
    C_valid = np.load(source_path + "C_valid_var.npy")

    n_input = 5

elif sys.argv[1] == "wiki":

    source_path = "../wiki/v8/"
    X_train = np.load(source_path + "X_train.npy")
    T_train = np.load(source_path + "T_train.npy")
    C_train = np.load(source_path + "C_train.npy")

    X_test = np.load(source_path + "X_test.npy")
    T_test = np.load(source_path + "T_test.npy")
    C_test = np.load(source_path + "C_test.npy")

    X_valid = np.load(source_path + "X_valid.npy")
    T_valid = np.load(source_path + "T_valid.npy")
    C_valid = np.load(source_path + "C_valid.npy")

    n_input = 8

else:
    print "parameter is not right, twitter or wiki."
    exit(0)

# The historical entry point selects on ten validation prefixes and
# reports the first five test prefixes.
n_classes = 1
before_steps = 10

num_units = 32
learning_rate = 0.001

out_weights = tf.Variable(tf.random_normal([num_units, n_classes]), trainable=True)
out_bias = tf.Variable(tf.random_normal([n_classes]), trainable=True)

X = tf.placeholder(tf.float32, shape=[None, None, n_input], name='X')
C = tf.placeholder(tf.float32, shape=[None,], name='C')
batch_size = tf.placeholder(tf.int32, name='batch_size')

# A causal GRU produces one nonnegative hazard for each input timestamp.
gru_cell = rnn.GRUCell(num_units)
outputs, _ = tf.nn.dynamic_rnn(cell=gru_cell, inputs=X, dtype="float32")

lambdas = tf.reshape(
    tf.nn.softplus(tf.matmul(tf.reshape(outputs, [-1, num_units]), out_weights) + out_bias),
    [batch_size, -1],
)

# The observed-event term is -log(1 - exp(-H)) and the censored term is H.
# Evaluating the event term with expm1 avoids exp(H) overflow and preserves
# precision for small H.
cumulative_hazard = tf.reduce_sum(lambdas, axis=1)
observed_event = tf.equal(C, 1.0)
# Keep censored rows out of log(0), including during backpropagation.
event_hazard = tf.where(observed_event, cumulative_hazard, tf.ones_like(cumulative_hazard))
event_loss = -tf.log(-tf.expm1(-event_hazard))
mle_loss = tf.reduce_mean(tf.where(observed_event, event_loss, cumulative_hazard))

# The upper-triangular mask sums hazards up to each prefix. Taking
# exp(-cumulative hazard) gives a nonincreasing survival curve.
mask = tf.placeholder(tf.float32, shape=[None, None], name='upper_matrix')
survivals = tf.exp(-tf.matmul(lambdas, mask))

# Preserve the original elementwise gradient clipping before Adam updates.
train_mle = tf.train.AdamOptimizer(learning_rate)
m_gvs = train_mle.compute_gradients(mle_loss)
m_capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in m_gvs]
train_mle_op = train_mle.apply_gradients(m_capped_gvs)

# Keep the checkpoint selected by validation accuracy for final testing.
saver = tf.train.Saver()
save_dir = '../checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "best_validation")

mini_batch_size = 16
mini_batch_size_test_valid = 5
# Legacy minibatch groups histories by observed length, so training needs no
# padding mask. It drops each length bucket's incomplete batch.
batch_x_train, _, _, batch_c_train = minibatch(
    X_train, T_train, C_train, mini_batch_size, n_input
)
batch_x_valid, batch_y_valid, _, _ = minibatch(
    X_valid, T_valid, C_valid, mini_batch_size_test_valid, n_input
)
batch_x_test, batch_y_test, batch_t_test, _ = minibatch(
    X_test, T_test, C_test, mini_batch_size_test_valid, n_input
)

best_validation_accuracy = 0.0
best_thrld = 0.0
last_improvement = 0
require_improvement = 20
num_epochs = 500

session = tf.Session()
session.run(tf.global_variables_initializer())

for epoch in range(num_epochs):
    for features, events in zip(batch_x_train, batch_c_train):
        _, loss = session.run(
            [train_mle_op, mle_loss],
            feed_dict={X: features, C: events, batch_size: len(features)},
        )

    if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
        # Scores are independent of the alert threshold. Cache each
        # validation batch once for this checkpoint and reuse it on the grid.
        validation_scores = []
        for features in batch_x_valid:
            sequence_length = features.shape[1]
            prefix_mask = upp_tri_mat(np.zeros((sequence_length, sequence_length)))
            scores = session.run(
                survivals,
                feed_dict={X: features, batch_size: len(features), mask: prefix_mask},
            )
            validation_scores.append(scores[:, :before_steps])

        # minibatch returns 1 for censored accounts, whereas the loss uses
        # 1 for observed events. A survival score above the cutoff predicts
        # the censored class.
        labels = np.concatenate(batch_y_valid)[:, None]
        scores = np.concatenate(validation_scores)
        threshold_scores = {
            threshold: np.mean(np.mean((scores > threshold) == labels, axis=0))
            for threshold in np.arange(0.1, 1.0, 0.02)
        }
        for threshold, accuracy in threshold_scores.items():
            if accuracy > best_validation_accuracy:
                best_validation_accuracy = accuracy
                best_thrld = threshold
                last_improvement = epoch + 1
                saver.save(sess=session, save_path=save_path)
                print "**** ", epoch + 1, best_validation_accuracy, best_thrld

    if epoch + 1 - last_improvement > require_improvement:
        break
    print "epoch: ", epoch, loss

saver.restore(sess=session, save_path=save_path)
test_prefixes = 5
event_predictions = []
event_labels = []
lead_times = defaultdict(list)

for features, labels, times in zip(batch_x_test, batch_y_test, batch_t_test):
    sequence_length = features.shape[1]
    prefix_mask = upp_tri_mat(np.zeros((sequence_length, sequence_length)))
    scores = session.run(
        survivals,
        feed_dict={X: features, batch_size: len(features), mask: prefix_mask},
    )
    predicted_censored = scores > best_thrld
    observed_events = labels == 0

    # Preserve the historical timing summary, including alerts at the final
    # timestamp. The fixed-prefix classification metrics below use five steps.
    if np.any(observed_events):
        event_histories = predicted_censored[observed_events]
        first_alerts = early_det(event_histories, np.zeros_like(event_histories))
        event_time = int(times[0])
        lead_times[event_time].extend(event_time - first_alerts)

    # Suspension is the positive class for precision, recall, and F1.
    event_predictions.append(~predicted_censored[:, :test_prefixes])
    event_labels.append(np.repeat(observed_events[:, None], test_prefixes, axis=1))

labels = np.concatenate(event_labels)
predictions = np.concatenate(event_predictions)
precision, recall, f1 = prec_reca_F1(labels, predictions)
accuracy = np.mean(labels == predictions, axis=0)

print
print
print
print "precision: ", precision
print "recall: ", recall
print "F1: ", f1
print "accuracy: ", accuracy
print "------------------------------------------------------"
for time, leads in lead_times.items():
    print time, ": ", np.mean(leads), len(leads)
