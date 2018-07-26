import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_log_error


class BaselineModel:
    label = "target"

    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

        self.mat = self.train_df.drop(self.label, axis=1).values
        self.labels = np.array(self.train_df[self.label].tolist())

        self.test_mat = self.test_df.drop(self.label, axis=1).values
        self.test_labels = np.array(self.test_df[self.label].tolist())

    def select_best_xgboost(self):
        # train xgboost
        kfold = KFold(5)
        i = 0

        models = []
        scores = []
        for train_index, valid_index in kfold.split(self.mat):
            train_mat = self.mat[train_index]
            valid_mat = self.mat[valid_index]

            train_labels = self.labels[train_index]
            valid_labels = self.labels[valid_index]

            train_dmatrix = xgb.DMatrix(train_mat, train_labels)
            valid_dmatrix = xgb.DMatrix(valid_mat, valid_labels)

            params = {
                'objective': 'reg:linear',
                'eval_metric': 'rmse',
                'eta': 0.001,
                'max_depth': 10,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'alpha': 0.001,
                'random_state': 42,
                'silent': True
            }

            watchlist = [(train_dmatrix, 'train'), (valid_dmatrix, 'valid')]
            model_xgb = xgb.train(params, train_dmatrix, 2000, watchlist, maximize=False, early_stopping_rounds=100,
                                  verbose_eval=100)

            test_dmatrix = xgb.DMatrix(self.test_mat)
            test_pred = model_xgb.predict(test_dmatrix, ntree_limit=model_xgb.best_ntree_limit)

            test_score = mean_squared_log_error(self.test_labels, test_pred)
            print("Best fold {0}: {1}".format(i, test_score))

            models.append(model_xgb)
            scores.append(test_score)

        min_score = 1000000
        model = None

        for i in range(0, len(scores)):
            if scores[i] < min_score:
                min_score = scores[i]
                model = models[i]

        self.model = model

    def predict_xgboost(self, test_df, output):
        ids = test_df['ID'].tolist()
        test_mat = test_df.drop('ID', axis=1).values

        test_dmatrix = xgb.DMatrix(test_mat)
        result = self.model.predict(test_dmatrix, ntree_limit=self.model.best_ntree_limit)

        result_df = pd.DataFrame(data={'ID': pd.Series(ids), 'target': pd.Series(result)})
        result_df.to_csv(output, index=False)


df = pd.read_csv('input/train.csv')
final_test_df = pd.read_csv('input/test.csv')

df.drop('ID', axis=1, inplace=True)

train_df, test_df = train_test_split(df, test_size=0.1)
model = BaselineModel(train_df, test_df)
model.select_best_xgboost()
model.predict_xgboost(final_test_df, "submission.csv")
