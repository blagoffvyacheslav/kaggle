import os
import numpy as np
import pandas as pd
import sklearn as skl
import xgboost as xgb


class MultiClassXGB(xgb.XGBClassifier):
    def set_classes(self, classes):
        self._le = skl.preprocessing.LabelEncoder().fit(classes)
        self.classes_ = list(self._le.classes_)
        self.n_classes_ = len(self.classes_)

        return self

    def set_eval(self, X_eval, y_eval, metric='mlogloss', early_stopping_rounds=50, verbose=10):
        self.eval_metric = metric
        self.eval_set = [(X_eval, self._le.transform(y_eval))]
        self.eval_verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        return self

    def set_model_file(self, name):
        self.xgb_model_name = name
        return self

    def fit(self, X, y, classes=None):
        evals_result = {}
        eval_metric = getattr(self, 'eval_metric', None)
        eval_set = getattr(self, 'eval_set', None)
        early_stopping_rounds = getattr(self, 'early_stopping_rounds', None)

        if not hasattr(self, 'classes_'):
            search_classes = classes if classes is not None else y

            if hasattr(search_classes, 'cat'):
                self.set_classes(search_classes.cat.categories)
            else:
                self.set_classes(search_classes)

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
            evals = list(xgb.DMatrix(x[0], label=x[1], missing=self.missing) for x in eval_set)
            nevals = len(evals)
            eval_names = ["validation_{}".format(i) for i in range(nevals)]
            evals = list(zip(evals, eval_names))
        else:
            evals = ()

        self._features_count = X.shape[1]

        training_labels = self._le.transform(y)
        train_dmatrix = xgb.DMatrix(X, label=training_labels, missing=self.missing)

        model_file = None
        xgb_model_name = getattr(self, 'xgb_model_name', None)
        if xgb_model_name:
            xgb_model_name = 'store/xgb_model_' + xgb_model_name
            if os.path.isfile(xgb_model_name):
                model_file = xgb_model_name

        verbose_eval = getattr(self, 'eval_verbose', True)
        self._Booster = xgb.train(xgb_options, train_dmatrix, self.n_estimators, evals=evals, evals_result=evals_result,
            feval=feval, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval, xgb_model=model_file)

        if xgb_model_name:
            self._Booster.save_model(xgb_model_name)

        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result_ = evals_result

        return self

    def plot_importance(self):
        return xgb.plot_importance(self.booster())

    def plot_tree(self, num_trees=2):
        return xgb.plot_tree(self.booster(), num_trees=num_trees)

    def predict_proba(self, data, output_margin=False, ntree_limit=0):
        proba = super(MultiClassXGB, self).predict_proba(data, output_margin, ntree_limit)
        return np.split(proba, 2, axis=1)[1].T


def get_model_proba(model, X, classes=None):
    proba = pd.DataFrame(model.predict_proba(X), index=X.index, columns=model.classes_)

    if classes is not None:
        proba_cl = pd.DataFrame(0, index=X.index, columns=classes)
        for cl in proba:
            if cl in proba_cl:
                proba_cl[cl] = proba[cl]

        proba = proba_cl

    return proba


def imbalance_log_loss(model, X, y, classes=None):
    if classes is None:
        le = skl.preprocessing.LabelEncoder()
        le.fit(y)
        classes = le.classes_

    proba = get_model_proba(model, X, classes)
    return skl.metrics.log_loss(y, proba.values)


def get_model_feature_scores(model, features, attr=None):
    if isinstance(features, pd.DataFrame):
        features = features.columns

    if attr is None:
        for a in ['feature_importances_', 'coef_']:
            if hasattr(model, a):
                attr = a
                break

    return pd.DataFrame({'val': getattr(model, attr)}, index=features).sort_values('val', ascending=False)
