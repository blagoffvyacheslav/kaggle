import numpy as np
from xgboost import XGBClassifier, DMatrix, train, plot_importance, plot_tree
from sklearn.preprocessing import LabelEncoder


class MultiClassXGB(XGBClassifier):
    def set_classes(self, classes):
        self._le = LabelEncoder().fit(classes)
        self.classes_ = list(self._le.classes_)
        self.n_classes_ = len(self.classes_)

        return self

    def set_eval(self, X_eval, y_eval, metric='mlogloss', early_stopping_rounds=10):
        self.eval_metric = metric
        self.eval_set = [(X_eval, self._le.transform(y_eval))]
        self.early_stopping_rounds = early_stopping_rounds
        return self

    def fit(self, X, y):
        evals_result = {}
        eval_metric = self.eval_metric
        eval_set = self.eval_set
        early_stopping_rounds = getattr(self, 'early_stopping_rounds')

        xgb_options = self.get_xgb_params()
        xgb_options["objective"] = "multi:softprob"
        xgb_options['num_class'] = self.n_classes_

        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                xgb_options.update({"eval_metric": eval_metric})

        if eval_set is not None:
            evals = list(DMatrix(x[0], label=x[1], missing=self.missing) for x in eval_set)
            nevals = len(evals)
            eval_names = ["validation_{}".format(i) for i in range(nevals)]
            evals = list(zip(evals, eval_names))
        else:
            evals = ()

        self._features_count = X.shape[1]

        training_labels = self._le.transform(y)
        train_dmatrix = DMatrix(X, label=training_labels, missing=self.missing)

        self._Booster = train(xgb_options, train_dmatrix, self.n_estimators,
            evals=evals, evals_result=evals_result, feval=feval, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result_ = evals_result

        return self

    def plot_importance(self):
        return plot_importance(self.booster())

    def plot_tree(self, num_trees=2):
        return plot_tree(self.booster(), num_trees=num_trees)

    def predict_proba(self, data, output_margin=False, ntree_limit=0):
        proba = super(MultiClassXGB, self).predict_proba(data, output_margin, ntree_limit)
        return np.split(proba, 2, axis=1)[1].T
