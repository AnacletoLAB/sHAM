import os
import pickle

import tensorflow as tf
import click

from data_utils.datahelper_noflag import *
from data_utils.datasets import DAVIS, KIBA
from sHAM import pruning, uCWS, uPWS
from sHAM import uUQ, uECSQ, pruning_uCWS, pruning_uPWS
from sHAM import pruning_uUQ, pruning_uECSQ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
exec(open("../GPU.py").read())

SEED = 1
from numpy.random import seed
seed(SEED)
tf.random.set_seed(SEED)

from tensorflow.python import _pywrap_util_port
print("MKL enabled:", _pywrap_util_port.IsMklEnabled())

@click.command()
@click.option('--compression', help='Type of compression')
@click.option('--net', help='original network datapath')
@click.option('--dataset', help='dataset')
@click.option('--learning_rate', default=0.0001, help='learning rate')
@click.option('--lr_cumulative', default=0.001, help='learning rate for cumulative gradient descent')
@click.option('--minibatch', default=64, help='size of minibatch')
@click.option('--prfc', default=0, help='percentage of pruned connection (dense layers)')
@click.option('--prcnn', default=0, help='percentage of pruned connection (convolutional layers)')
@click.option('--clusterfc', default=0, help='different values for all dense layers')
@click.option('--clustercnn', default=0, help='different values for all dense layers')
@click.option('--tr', default=0.001, help='treshold for ECSQ')
@click.option('--lambd', default=0., help='coefficient for entrophy with ECSQ')
@click.option('--logger', default=False, help='set True for logging train into txt')
@click.option('--ptnc', default=0, help='patience (defualt 0)')

# This script does not excercise old non-unified methods. Check https://github.com/giosumarin/ICPR2020_sHAM for those
def main(compression, net, dataset, learning_rate, lr_cumulative, minibatch, prfc, prcnn, clusterfc, clustercnn, tr, lambd, logger, ptnc):

    # Load model
    model = tf.keras.models.load_model(net)

    # Compression definition
    if compression == 'pr':
        compression_model = pruning.pruning(model=model, perc_prun_for_dense=prfc, perc_prun_for_cnn=prcnn)
    elif compression == 'uCWS':
        compression_model = uCWS.uCWS(model=model, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc)
    elif compression == 'uPWS':
        compression_model = uPWS.uPWS(model=model, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc)
    elif compression == 'uUQ':
        compression_model = uUQ.uUQ(model=model, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc)
    elif compression == 'uECSQ':
        compression_model = uECSQ.uECSQ(model=model, clusters_for_conv_layers=3*clustercnn, clusters_for_dense_layers=3*clusterfc, wanted_clusters_cnn=clustercnn, wanted_clusters_fc=clusterfc, tr=tr, lamb=lambd)
    elif compression == 'pruCWS':
        compression_model = pruning_uCWS.pruning_uCWS(model=model, perc_prun_for_dense=prfc, perc_prun_for_cnn=prcnn, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc)
    elif compression == 'pruPWS':
        compression_model = pruning_uPWS.pruning_uPWS(model=model, perc_prun_for_dense=prfc, perc_prun_for_cnn=prcnn, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc)
    elif compression == 'pruUQ':
        compression_model = pruning_uUQ.pruning_uUQ(model=model, perc_prun_for_dense=prfc, perc_prun_for_cnn=prcnn, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc)
    elif compression == 'pruECSQ':
        compression_model = pruning_uECSQ.pruning_uECSQ(model=model, perc_prun_for_dense=prfc, perc_prun_for_cnn=prcnn, clusters_for_conv_layers=3*clustercnn, clusters_for_dense_layers=3*clusterfc, wanted_clusters_cnn=clustercnn, wanted_clusters_fc=clusterfc, tr=tr, lamb=lambd)

    # Load dataset
    if dataset == "DAVIS":
        dataset, x_train, y_train, x_test, y_test = DAVIS(minibatch)
    elif dataset == "KIBA":
        dataset, x_train, y_train, x_test, y_test = KIBA(minibatch)

    # Pre-compression prediction assessment
    pre_compr_train = compression_model.model.evaluate(x_train, y_train)
    pre_compr_test = compression_model.model.evaluate(x_test, y_test)
    print("before compression, performance on train -->", pre_compr_train)
    print("before compression, performance on test -->", pre_compr_test)

    # Model compression
    if compression == 'pr':
        compression_model.apply_pruning()
    elif compression == 'uCWS':
        compression_model.apply_uCWS()
    elif compression == 'uPWS':
        compression_model.apply_uPWS()
    elif compression == 'uUQ':
        compression_model.apply_uUQ()
    elif compression == 'uECSQ':
        lambdas = [0.05, 0.04, 0.03, 0.02, 0.01, 0.008, 0.0065, 0.005, 0.001, 0.0008, 0.0005, 0.0002, 0.0001]
        compression_model.tune_lambda(lambdas)
        compression_model.apply_uECSQ()
    elif compression == 'pruCWS':
        compression_model.apply_pr_uCWS()
    elif compression == 'pruPWS':
        compression_model.apply_pr_uPWS()
    elif compression == 'pruUQ':
        compression_model.apply_pr_uUQ()
    elif compression == 'pruECSQ':
        lambdas = [0.05, 0.04, 0.03, 0.02, 0.01, 0.008, 0.0065, 0.005, 0.001, 0.0008, 0.0005, 0.0002, 0.0001]
        compression_model.tune_lambda(lambdas)
        compression_model.apply_pr_uECSQ()

    # Post-compression prediction assessment
    compression_model.set_loss(tf.keras.losses.MeanSquaredError())
    compression_model.set_optimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate))
    post_compr_train = compression_model.model.evaluate(x_train, y_train)
    post_compr_test = compression_model.model.evaluate(x_test, y_test)
    print("Setting initial compression setting before retraining, performance on train -->" , post_compr_train)
    print("Setting initial compression setting before retraining, performance on test -->" , post_compr_test)

    # Model re-train
    if compression == "pr":
        compression_model.train_pr_deepdta(epochs=20, dataset=dataset, X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, step_per_epoch=10000000, patience=ptnc)
    else:
        compression_model.train_ws(epochs=50, lr=lr_cumulative, dataset=dataset, X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, patience=ptnc)

    # Model save
    name_net = (net.split("/")[-1])[:-3]
    TRAIN_RES = ([pre_compr_train] + [post_compr_train] + compression_model.acc_train)
    TEST_RES = ([pre_compr_test] + [post_compr_test] + compression_model.acc_test)

    if compression in ['uUQ', 'pruUQ', 'uECSQ', 'pruECSQ']:
        compression_param = "-".join([str(x) for x in [prfc, prcnn, len(compression_model.centers_fc), len(compression_model.centers_cnn)]]) + '-'
    else:
        compression_param = "-".join([str(x) for x in [prfc, prcnn, clusterfc, clustercnn]]) + '-'

    if logger:
        with open("{}_{}.txt".format(name_net, compression), "a+") as tex:
            tex.write("lr {} {} -->\n {}\n , {}\n\n".format(learning_rate, compression_param, TRAIN_RES, TEST_RES))

    DIR="{}/{}".format(name_net, compression)
    TO_SAVE = "{}{}".format(compression_param, round(TEST_RES[-1],5))
    if not os.path.isdir(name_net):
        os.mkdir(name_net)
    if not os.path.isdir(DIR):
        os.mkdir(DIR)

    end_weights = compression_model.model.get_weights()

    with open(DIR+"/"+TO_SAVE+".h5", "wb") as file:
        pickle.dump(end_weights, file)

if __name__ == '__main__':
    main()
