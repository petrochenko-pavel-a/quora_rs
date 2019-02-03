from model import *
from workspace import *

from utils import *

import pandas as pd
import os
import yaml
import argparse
import time

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

def load_config(path,prepocessing_path,DIR):
    EMB_DIR = DIR + "input/embeddings/"
    train_dataset = SimpleCsvDataSet(DIR + '/input/train.csv', "question_text", "target")
    test_dataset = TestCsvDataSet(DIR + '/input/test.csv', "question_text")
    config = load_yaml(path)
    cd=os.getcwd()
    os.chdir(prepocessing_path)
    workspace=ClassificationWorkspace(config,EMB_DIR,path)
    workspace.prepare(train_dataset, test_dataset)
    os.chdir(cd)
    return workspace

class BranchController:
    def __init__(self, weightsFolderPath, branchesToLayers, workspaceConfig, foldNumber):
        self.weightsFolderPath = weightsFolderPath
        self.branchesToLayers = branchesToLayers
        self.workspaceConfig = workspaceConfig
        self.foldNumber = foldNumber

    def foundBestEpoch(self, epochNumber):
        for branchName in self.workspaceConfig["branches"]:
            branchConfig = self.workspaceConfig["branches"][branchName]
            if "saveWeights" in branchConfig:
                fileName = self.branchFileName(branchName)
                branchDictionary = self.branchesToLayers[branchName]
                if branchDictionary is not None:
                    layerToWeights = {}
                    for layerName in branchDictionary:
                        layer = branchDictionary[layerName]
                        if not self.ignoreEmbeddings(branchName) or not isinstance(layer, keras.layers.Embedding):
                            layerToWeights[layerName] = layer.get_weights()
                    save(self.weightsFolderPath + "/" + fileName, layerToWeights)

    def epochEnd(self, epochNumber):
        return

    def epochStart(self, epochNumber):
        for branchName in self.workspaceConfig["branches"]:
            branchConfig = self.workspaceConfig["branches"][branchName]
            if "freeze" in branchConfig:
                freezeSetting = branchConfig["freeze"]
                freezeStart = freezeSetting[0]
                freezeEnd = freezeSetting[1]
                if epochNumber>=freezeStart and epochNumber < freezeEnd:
                    print("----------Branch Controller Freezing branch " + branchName + " on epoch " + str(epochNumber))
                    self.freezeBranch(branchName)
                else:
                    print("----------Branch Controller Unfreezing branch " + branchName + " on epoch " + str(epochNumber))
                    self.unfreezeBranch(branchName)

    def freezeBranch(self, branchName):
        branchConfig = self.workspaceConfig["branches"][branchName]

        branchDictionary = self.branchesToLayers[branchName]
        if branchDictionary is not None:
            for layerName in branchDictionary:
                layer = branchDictionary[layerName]
                if not isinstance(layer, keras.layers.Embedding) or ("freezeEmbedding" in branchConfig and branchConfig["freezeEmbedding"]):
                    layer.trainable = False

    def unfreezeBranch(self, branchName):
        branchConfig = self.workspaceConfig["branches"][branchName]

        branchDictionary = self.branchesToLayers[branchName]
        if branchDictionary is not None:
            for layerName in branchDictionary:
                layer = branchDictionary[layerName]
                if not isinstance(layer, keras.layers.Embedding) or ("freezeEmbedding" in branchConfig and branchConfig["freezeEmbedding"]):
                    layer.trainable = True

    def branchFileName(self, branchName):
        return "fold_" + str(self.foldNumber) + "_branch_" + branchName + ".weights"

    def loadWeights(self):
        for branchName in self.workspaceConfig["branches"]:
            branchConfig = self.workspaceConfig["branches"][branchName]
            if "loadWeights" in branchConfig:
                fileName = self.branchFileName(branchName)
                branchDictionary = self.branchesToLayers[branchName]
                if branchDictionary is not None:
                    fullFileName = self.weightsFolderPath + "/" + fileName
                    if os.path.exists(fullFileName):
                        print("----------Branch Controller loading weights")
                        layerToWeights = load(fullFileName)
                        for layerName in layerToWeights:
                            layerWeights = layerToWeights[layerName]
                            layer = branchDictionary[layerName]
                            layer.set_weights(layerWeights)
                        print("----------Branch Controller finished loading weights")

    def ignoreEmbeddings(self, branchName):
        return True
class UpgModelCheckpoint(keras.callbacks.Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, branchController:BranchController = None):
        super(UpgModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.branchController:BranchController = branchController

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_begin(self, epoch, logs=None):
        if self.branchController is not None:
            self.branchController.epochStart(epoch)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.branchController is not None:
            self.branchController.epochEnd(epoch)

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if  self.branchController is not None:
                            self.branchController.foundBestEpoch(epoch)

                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)



def evalModel(workspace:ClassificationWorkspace,gpu):
    modelStartTime = time.time()

    best_tresholds = []
    best_scores = []

    combined_tresh = []
    combined_scores = []
    dir=os.path.dirname(workspace.path)
    weights_folder=dir+"/weights"
    ensure_exists(weights_folder)

    holdout_best=[]
    holdout_blend=[]
    pred_holdout,holdout_data=workspace.get_holdout()
    folds_to_calculate = workspace.fold_count
    if workspace.folds_to_calculate: folds_to_calculate = workspace.folds_to_calculate

    perEpochTime = 0.0
    for foldNum in range(folds_to_calculate):

        with K.tf.device('/gpu:'+str(gpu)):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

            sess = tf.Session(config=config)
            K.set_session(sess)
            modelCreationResults = create_model_from_yaml(workspace)
            m = modelCreationResults[0]
            branchesToLayers = modelCreationResults[1]

            pred_train, pred_val, train_, val_ = workspace.get_data(foldNum)
            weights_path = weights_folder + "/fold_best_" + str(foldNum) + ".weights"
            branchController = BranchController(weights_folder, branchesToLayers,
                                                workspace.config["model"], foldNum)
            checkpoint=UpgModelCheckpoint(weights_path, save_best_only=True,
                                                       monitor="val_acc",
                                          mode="max", save_weights_only=True,
                                          branchController = branchController)
            branchController.loadWeights()

            fitStartTime = time.time()
            m.fit(train_, pred_train, workspace.config["batch"], workspace.config["epochs"], validation_data=(val_,pred_val), shuffle=True,
                  callbacks=[checkpoint], verbose=1)
            fitEndTime = time.time()
            perEpochTime += (fitEndTime - fitStartTime)/(folds_to_calculate*workspace.config["epochs"])
            m.save_weights(weights_folder+"/fold_last_" + str(foldNum)+".weights")
            pred_last = m.predict(val_,batch_size=workspace.config["batch"])

            holdout_last_pred = m.predict(holdout_data, batch_size=workspace.config["batch"])
            m.load_weights(weights_path)

            pred_best = m.predict(val_,batch_size=workspace.config["batch"])


            treshold_best=eval_treshold_and_score(pred_best, pred_val)
            treshold_snaphot = eval_treshold_and_score((pred_best+pred_last) / 2, pred_val)
            treshold1_last = eval_treshold_and_score(pred_last, pred_val)

            best_tresholds.append(treshold_best[0])
            best_scores.append(treshold_best[1])

            combined_tresh.append(treshold_snaphot[0])
            combined_scores.append(treshold_snaphot[1])

            holdout_pred=m.predict(holdout_data,batch_size=workspace.config["batch"])
            treshold_h = eval_treshold_and_score(holdout_pred, pred_holdout)

            os.remove(weights_path)
            os.remove(weights_folder+"/fold_last_" + str(foldNum)+".weights")


            print("Fold:",foldNum,treshold_best[0],treshold_best[1]," blending last and best:",treshold_snaphot[0],treshold_snaphot[1]," Last:",treshold1_last[0],treshold1_last[1],"Holdout:",treshold_h[0],treshold_h[1])

            metrics={
                "fold": int(foldNum),
                "treshold_best":float(treshold_best[0]),
                "score_best":float(treshold_best[1]),
                "treshold_snaphot":float(treshold_snaphot[0]),
                "score_snaphot": float(treshold_snaphot[1]),
                "treshold_last": float(treshold1_last[0]),
                "score_last": float(treshold1_last[1]),
                "treshold_holdout": float(treshold_h[0]),
                "score_holdout": float(treshold_h[1]),
            }
            with open(dir+"/metrics"+str(foldNum)+".yaml","w",encoding="utf8") as f:
                yaml.dump(metrics,f,default_flow_style=False)

            holdout_best.append(holdout_pred)
            holdout_blend.append(holdout_pred)
            holdout_blend.append(holdout_last_pred)
            K.clear_session()

            del m
            del sess
            del branchesToLayers
            del branchController
            del checkpoint
            del modelCreationResults
            gc.collect()

    average_preds=np.array(holdout_best).mean(axis=0)
    blend_preds = np.array(holdout_blend).mean(axis=0)
    holdout_tresh = eval_treshold_and_score(average_preds, pred_holdout)
    holdout_tresh_blend = eval_treshold_and_score(blend_preds, pred_holdout)
    blendAver=eval_f1_score(blend_preds,pred_holdout,np.mean(np.array(combined_tresh))*0.93)
    print("Evaluating all folds blend:",holdout_tresh[0],holdout_tresh[1],holdout_tresh_blend[0],holdout_tresh_blend[1],"Av Blend:",blendAver)

    modelEndTime = time.time()
    modelTotalTime = modelEndTime - modelStartTime
    perFoldTime = modelTotalTime / folds_to_calculate

    metrics = {
        "holdout_blend_treshold": float(holdout_tresh[0]),
        "holdout_blend_score": float(holdout_tresh[1]),

        "best_treshold_average": float(np.mean(np.array(best_tresholds))),
        "best_score_average": float(np.mean(np.array(best_scores))),
        "snapshot_treshold_average": float(np.mean(np.array(combined_tresh))),
        "snapshot_score_average": float(np.mean(np.array(combined_scores))),

        "holdout_super_blend_treshold": float(holdout_tresh_blend[0]),
        "holdout_super_blend_score": float(holdout_tresh_blend[1]),
        "Blend with averaged tresh:": float(blendAver),
        "Total work time:": modelTotalTime,
        "Time per fold:": perFoldTime,
        "Time per epoch:": perEpochTime
    }
    with open(dir+"/metrics_final" + ".yaml", "w", encoding="utf8") as f:
        yaml.dump(metrics, f,default_flow_style=False)
    return np.mean(np.array(best_tresholds)),np.mean(np.array(best_scores)),np.mean(np.array(combined_tresh)),np.mean(np.array(combined_scores)),holdout_tresh[0],holdout_tresh[1]


def generate_submition(model, workspace:ClassificationWorkspace, treshold):
    preds=model.predict(workspace.get_test())

    submission = pd.read_csv('submission.csv')
    pred_3 = []
    for pred_scalar in preds:
        pred_3.append(int(pred_scalar > treshold))
    submission['prediction'] = pred_3
    submission.to_csv('submission.csv', index=False)

    sbase = "".join(chr(i) for i in range(0,65535) if chr(i).isalnum())
    pred_str = ''.join(str(x) for x in list(submission.prediction.values))
    pred_compress_copy= encode(pred_str, sbase)
    with open("encoded.txt","w",encoding="utf8") as f:
        f.write(pred_compress_copy)

from keras.layers import *
import keras
def main():
    parser = argparse.ArgumentParser(description='Train quora')
    parser.add_argument('--data', type=str, default=".",
                        help='folder to store preprocessing data')
    parser.add_argument('--input', type=str, default="config.yaml",
                        help='path to config')
    parser.add_argument('--quora', type=str, default="",
                        help='path to quora')
    parser.add_argument('--gpu', type=int, default=0,
                        help='path to quora')
    args = parser.parse_args()


    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # args.gpu=0

    #
    if os.path.isdir(args.input):
        for v in os.listdir(args.input):
            ps=args.input + "/" + v + "/config.yaml";
            print("Handling " + ps)
            if os.path.exists(ps):
                evalModel(load_config(ps, args.data, args.quora + "/"), args.gpu)
        exit(0)
    evalModel(load_config(args.input, args.data,args.quora+"/"),args.gpu)
    # exit(0)
    # for i in range(1,3):
    #     for j in range(15):
    #         try:
    #             evalModel(load_config("D:/quora/word_eval"+str(i)+"/exp"+str(j)+"/config.yaml", args.data,args.quora+"/"),args.gpu)
    #         except:
    #             pass
    pass
if __name__ == '__main__':
    main()
