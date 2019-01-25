import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold,train_test_split



class SimpleCsvDataSet:

    def __init__(self,dataframe,textColumn,targetColumn):
        if isinstance(dataframe,str):
            dataframe=pd.read_csv(dataframe)
        self.text=dataframe[textColumn].values
        self.target = dataframe[targetColumn].values

    def __getitem__(self, item):
        return self.text[item],self.target[item]

    def __len__(self):
        return len(self.text)

    def texts(self):
        return self.text

    def predictions(self):
        return self.target


class TestCsvDataSet:

    def __init__(self,dataframe,textColumn):
        if isinstance(dataframe,str):
            dataframe=pd.read_csv(dataframe)
        self.text=dataframe[textColumn].values

    def __getitem__(self, item):
        return self.text[item],0

    def __len__(self):
        return len(self.text)

    def texts(self):
        return self.text


class KFoldDataSet:
    def __init__(self,ds,indexes,foldCount=5,random=43,startified=True,):
        self.ds=ds;
        if startified:
            preds=ds.predictions()
            self.indexes=[i for i in StratifiedKFold(foldCount, True, random).split(indexes, preds[indexes])]
        else: self.indexes=[i for i in KFold(foldCount, True, random).split(indexes)]

def holdout(ds,part,rand):
    return train_test_split(range(len(ds)),test_size=part,stratify=ds.predictions(),random_state=rand)