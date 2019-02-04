from model import *
from workspace import *

from utils import *

import pandas as pd
import os
import yaml
import argparse
import gc



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


class RecordOnTest(keras.callbacks.Callback):

    def __init__(self,model,datas,all):
        self.model=model
        self.datas=datas
        self.all=all

    def on_epoch_end(self, epoch, logs=None):
        preds=[]
        if epoch>-1:
            print("AA")
            for d in self.datas:
                preds.append(self.model.predict(d, batch_size=1024))
            self.all.append(preds)
        pass

from keras.backend.common import set_floatx
from keras.callbacks import *


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())
#set_floatx('float16')
from matplotlib import pyplot as plt
import math
class LRFinder:
    """
    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """
    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if math.isnan(loss) or loss > self.best_loss * 4:
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1):
        num_batches = epochs * x_train[0].shape[0] / batch_size
        self.lr_mult = (end_lr / start_lr) ** (1 / num_batches)

        # Save weights into a file
        self.model.save_weights('tmp.h5')

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=[callback])

        # Restore the weights to the state before model fitting
        self.model.load_weights('tmp.h5')

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale('log')

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivative = (self.losses[i] - self.losses[i - sma]) / sma
            derivatives.append(derivative)

        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], derivatives[n_skip_beginning:-n_skip_end])
        plt.xscale('log')
        plt.ylim(y_lim)
def evalModel(workspace:ClassificationWorkspace,gpu):
    best_tresholds = []
    best_scores = []

    combined_tresh = []
    combined_scores = []
    dir=os.path.dirname(workspace.path)
    weights_folder=dir+"/weights"
    ensure_exists(weights_folder)

    holdout_best=[]
    holdout_blend=[]
    test_blend=[]
    pred_holdout,holdout_data=workspace.get_holdout()
    workspace.get_test()
    holdOutPredictions=[]



    for foldNum in range(workspace.fold_count):
        with K.tf.device('/gpu:'+str(gpu)):
            K.set_session(K.tf.Session())
            allData=[]


            m = create_model_from_yaml(workspace,foldNum,0)

            pred_train, pred_val, train_, val_ = workspace.get_data(foldNum)
            rs = RecordOnTest(m, [val_, holdout_data, workspace.get_test()],allData);


            m.fit(train_, pred_train, workspace.config["batch"], 2, validation_data=(val_,pred_val), shuffle=True, callbacks=[], verbose=2)

            #train_ = workspace.augment(train_)

            # m.fit(train_, pred_train, workspace.config["batch"], 2, validation_data=(val_, pred_val), shuffle=True,
            #       callbacks=[], verbose=1)

            predicted=m.predict(train_,batch_size=1024)

            hold_predicted = m.predict(holdout_data, batch_size=1024)

            validation=m.predict(val_,batch_size=1024)


            r=set(list(np.where(predicted>0.01)[0]))
            rsa=np.array(list(r.union(list(np.where(pred_train>0))[0])))

            t1 = train_[0][rsa]
            t2 = train_[1][rsa]
            m.fit([t1, t2, train_[2][rsa]], pred_train[rsa], workspace.config["batch"], 3,
                  validation_data=(val_, pred_val), shuffle=True,
                  callbacks=[rs], verbose=2)

            t1=workspace.augment([train_[0][rsa],train_[1][rsa],train_[2][rsa]])

            m.fit(t1, pred_train[rsa], workspace.config["batch"], 3, validation_data=(val_, pred_val), shuffle=True,
                  callbacks=[rs], verbose=2)

            validation1=m.predict(val_,batch_size=1024)

            validation1[np.where(validation<0.01)[0]]=0



            allh = [r[0] for r in rs.all]
            validation1 = np.mean(np.array(allh), axis=0)
            validation1[np.where(validation < 0.01)[0]] = 0
            treshold_best = eval_treshold_and_score(validation1, pred_val)
            print("AAA2", treshold_best)
            # i=Input((2,))
            # d = Dense(5, activation="relu")(i)
            # d = Dense(1,activation="sigmoid")(d)
            # ma=keras.Model(i,d)
            #ma.compile("adam","binary_crossentropy",["accuracy"])
            #ma.fit(np.concatenate([validation,validation1],axis=1),pred_val,epochs=10,batch_size=100,validation_split=0.2,verbose=1)

            # for i in range(100):
            #     print(i,pred_val[np.where(predicted<i*0.01)].sum())

            #m.save_weights(weights_folder+"/fold_last_" + str(foldNum)+".weights")
            #pred_last = m.predict(val_,batch_size=workspace.config["batch"])

            #mz=keras.Model(m.inputs,m.layers[-2].output)
            #foldLastPredictions=mz.predict(holdout_data,batch_size=workspace.config["batch"])
            #holdOutPredictions.append(foldLastPredictions)

            #holdout_last_pred = m.predict(holdout_data, batch_size=workspace.config["batch"])
            #m.load_weights(weights_path)


            # pred_best = m.predict(val_,batch_size=workspace.config["batch"])
            #
            # mz = keras.Model(m.inputs, m.layers[-2].output)
            # foldBestPredictions = mz.predict(holdout_data, batch_size=workspace.config["batch"])
            # holdOutPredictions.append(foldBestPredictions)
            #

            allPreds = [r[0] for r in rs.all]

            for j in range(2):
                allVal = [r[j] for r in rs.all]
                z=np.mean(np.array(allVal),axis=0)

                p=pred_val
                if j==1:
                    p=pred_holdout

                treshold_best=eval_treshold_and_score(z, p)
                print(j,treshold_best)
            allh = [r[1] for r in rs.all]
            ht = np.mean(np.array(allh), axis=0)
            ht[np.where(hold_predicted<0.01)[0]]=0
            #hf=(allVal[1]+allVal[2]+allVal[3]+allVal[4])/4
            holdout_blend.append(ht)

            allTest = [r[2] for r in rs.all]
            ht = np.mean(np.array(allTest),axis=0)
            test_blend.append(ht)

            blend_preds = np.array(holdout_blend).mean(axis=0)
            blendAver = eval_treshold_and_score(blend_preds, pred_holdout)
            print(blendAver)

            # treshold_snaphot = eval_treshold_and_score((pred_best+pred_last) / 2, pred_val)
            # treshold1_last = eval_treshold_and_score(pred_last, pred_val)
            #
            # best_tresholds.append(treshold_best[0])
            # best_scores.append(treshold_best[1])
            #
            # combined_tresh.append(treshold_snaphot[0])
            # combined_scores.append(treshold_snaphot[1])
            #
            # holdout_pred=m.predict(holdout_data,batch_size=workspace.config["batch"])
            # treshold_h = eval_treshold_and_score(holdout_pred, pred_holdout)
            #
            # os.remove(weights_path)
            # os.remove(weights_folder+"/fold_last_" + str(foldNum)+".weights")


            #print("Fold:",foldNum,treshold_best[0],treshold_best[1]," blending last and best:",treshold_snaphot[0],treshold_snaphot[1]," Last:",treshold1_last[0],treshold1_last[1],"Holdout:",treshold_h[0],treshold_h[1])

            # metrics={
            #     "fold": int(foldNum),
            #     "treshold_best":float(treshold_best[0]),
            #     "score_best":float(treshold_best[1]),
            #     "treshold_snaphot":float(treshold_snaphot[0]),
            #     "score_snaphot": float(treshold_snaphot[1]),
            #     "treshold_last": float(treshold1_last[0]),
            #     "score_last": float(treshold1_last[1]),
            #     "treshold_holdout": float(treshold_h[0]),
            #     "score_holdout": float(treshold_h[1]),
            # }
            # with open(dir+"/metrics"+str(foldNum)+".yaml","w",encoding="utf8") as f:
            #     yaml.dump(metrics,f,default_flow_style=False)

            # holdout_best.append(holdout_pred)
            # holdout_blend.append(holdout_pred)
            #holdout_blend.append(holdout_last_pred)
            K.clear_session()





    # st=eval_treshold_and_score(stackPred, pred_holdout[f.indexes[0][1]])
    # print(st)


    #average_preds=np.array(holdout_best).mean(axis=0)
    blend_preds = np.array(holdout_blend).mean(axis=0)
    blend_test = np.array(test_blend).mean(axis=0)
    #holdout_tresh = eval_treshold_and_score(average_preds, pred_holdout)
    #holdout_tresh_blend = eval_treshold_and_score(blend_preds, pred_holdout)
    blendAver=eval_treshold_and_score(blend_preds, pred_holdout)
    print(blendAver)
    generate_submition(blend_test,blendAver[0])

    #print("Evaluating all folds blend:",holdout_tresh[0],holdout_tresh[1],holdout_tresh_blend[0],holdout_tresh_blend[1],"Av Blend:",blendAver)

    metrics = {
         "holdout_blend_treshold": float(blendAver[0]),
         "holdout_blend_score": float(blendAver[1]),

         #"best_treshold_average": float(np.mean(np.array(best_tresholds))),
         #"best_score_average": float(np.mean(np.array(best_scores))),
         #"snapshot_treshold_average": float(np.mean(np.array(combined_tresh))),
         #"snapshot_score_average": float(np.mean(np.array(combined_scores))),

         #"holdout_super_blend_treshold": float(holdout_tresh_blend[0]),
         #"holdout_super_blend_score": float(holdout_tresh_blend[1]),
         #"Blend with averaged tresh:": float(blendAver),
     }
    with open(dir+"/metrics_final" + ".yaml", "w", encoding="utf8") as f:
         yaml.dump(metrics, f,default_flow_style=False)
    return blendAver


def generate_submition( preds, treshold):
    #preds=model.predict(workspace.get_test())

    submission = pd.read_csv('"D:/quora/input/sample_submission.csv"')
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


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.gpu=0

    #
    if os.path.isdir(args.input):
        for v in os.listdir(args.input):
            ps=args.input + "/" + v + "/config.yaml";
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
