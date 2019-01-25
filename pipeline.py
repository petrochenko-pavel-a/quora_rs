from model import *
from workspace import *
from keras.layers import *
from utils import *
import keras
import pandas as pd


DIR="D:/quora/"
EMB_DIR=DIR+"input/embeddings/"
WEIGHTS_FOLDER="D:/quora/weights/"

train_dataset = SimpleCsvDataSet(DIR+'/input/train.csv', "question_text", "target")
test_dataset = TestCsvDataSet(DIR+'/input/test.csv', "question_text")

config = load_yaml("./exp/config.yaml")
os.chdir("exp")
workspace=ClassificationWorkspace(config,EMB_DIR)
workspace.prepare(train_dataset,test_dataset)


class CFG:
    def __init__(self):
        self.dropout=np.random.rand()
        self.epochs=round(np.random.randint(3,7))
        self.dropout1=np.random.rand()/2
        self.count=round(np.random.randint(1,7))
        self.addGlobal=np.random.rand()>0.5
        self.addChars = np.random.rand() > 0.5
        self.spatial=np.random.rand() > 0.5
        pass

    def __str__(self):
        return str(self.dropout)+","+str(self.dropout1)+","+str(self.epochs)+","+str(self.count)+","+str(self.addGlobal)+","+str(self.addChars)+","+str(self.spatial)


def evalModel(cfg,modelCreator):
    best_tresholds = []
    best_scores = []

    combined_tresh = []
    combined_scores = []

    combined=[]
    pred_holdout,holdout_data=workspace.get_holdout()
    for foldNum in range(workspace.fold_count):
        K.set_session(K.tf.Session())
        m = modelCreator(cfg,workspace)

        pred_train, pred_val, train_, val_ = workspace.get_data(foldNum)

        m.fit(train_, pred_train, 1024, cfg.epochs, validation_data=(val_,pred_val), shuffle=True, callbacks=[keras.callbacks.ModelCheckpoint(WEIGHTS_FOLDER+"fold_" + str(cfg) + "_" + str(foldNum), save_best_only=True, monitor="val_acc", mode="max")], verbose=1)

        pred_last = m.predict(val_)

        m.load_weights(WEIGHTS_FOLDER+"fold_"+str(cfg)+"_"+str(foldNum))

        pred_best = m.predict(val_)

        treshold_best=eval_treshold_and_score(pred_best, pred_val)
        treshold_snaphot = eval_treshold_and_score((pred_best+pred_last) / 2, pred_val)
        treshold1_last = eval_treshold_and_score(pred_last, pred_val)

        best_tresholds.append(treshold_best[0])
        best_scores.append(treshold_best[1])

        combined_tresh.append(treshold_snaphot[0])
        combined_scores.append(treshold_snaphot[1])

        holdout_pred=m.predict(holdout_data)
        treshold_h = eval_treshold_and_score(holdout_pred, pred_holdout)

        print("Fold:",foldNum,treshold_best[0],treshold_best[1]," blending last and best:",treshold_snaphot[0],treshold_snaphot[1]," Last:",treshold1_last[0],treshold1_last[1],"Holdout:",treshold_h[0],treshold_h[1])

        combined.append(holdout_pred)
        K.clear_session()

    averagePreds=np.array(combined).mean(axis=0)
    holdout_tresh = eval_treshold_and_score(averagePreds, pred_holdout)
    print("Evaluating all folds blend:",holdout_tresh[0],holdout_tresh[1])
    return np.mean(np.array(best_tresholds)),np.mean(np.array(best_scores)),np.mean(np.array(combined_tresh)),np.mean(np.array(combined_scores)),holdout_tresh[0],holdout_tresh[1]


def eval_random_models(count:int):
    with open("results.txt","w") as scoreFile:
        for i in range(count):
            c=CFG()
            res=evalModel(c,create_model)
            scoreFile.write(str(c)+","+",".join(str(x) for x in res))


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


# Lets train and eval model
c=CFG()
c.dropout=0.3
c.dropout1=0.2
c.epochs=1
c.count=4
c.spatial=True
c.addGlobal=True
c.addChars=True
evalModel(c,create_model)