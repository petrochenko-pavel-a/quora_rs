from model import *
from workspace import *

from utils import *

import pandas as pd
import os
import yaml
import argparse



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
    pred_holdout,holdout_data=workspace.get_holdout()
    for foldNum in range(workspace.fold_count):
        with K.tf.device('/gpu:'+str(gpu)):
            K.set_session(K.tf.Session())
            m = create_model_from_yaml(workspace)

            pred_train, pred_val, train_, val_ = workspace.get_data(foldNum)
            weights_path = weights_folder + "/fold_best_" + str(foldNum) + ".weights"
            checkpoint=keras.callbacks.ModelCheckpoint(weights_path, save_best_only=True,
                                                       monitor="val_acc", mode="max", save_weights_only=True)
            m.fit(train_, pred_train, workspace.config["batch"], workspace.config["epochs"], validation_data=(val_,pred_val), shuffle=True,
                  callbacks=[checkpoint], verbose=1)
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

    average_preds=np.array(holdout_best).mean(axis=0)
    blend_preds = np.array(holdout_blend).mean(axis=0)
    holdout_tresh = eval_treshold_and_score(average_preds, pred_holdout)
    holdout_tresh_blend = eval_treshold_and_score(blend_preds, pred_holdout)
    blendAver=eval_f1_score(blend_preds,pred_holdout,np.mean(np.array(combined_tresh))*0.93)
    print("Evaluating all folds blend:",holdout_tresh[0],holdout_tresh[1],holdout_tresh_blend[0],holdout_tresh_blend[1],"Av Blend:",blendAver)

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
