from data import np, pd, skl, xgb, os, sys, itertools
from ipython import FloatProgress, display


class LogisticXGB(xgb.XGBClassifier):
    def _transform_y(self, y):
        return y

    def set_eval(self, X_eval, y_eval, metric='logloss', early_stopping_rounds=50, verbose=10):
        self.eval_metric = metric
        self.eval_set = [(X_eval, self._transform_y(y_eval))]
        self.eval_verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        return self

    def set_model_file(self, name):
        self.xgb_model_name = name
        return self

    def set_classes(self, classes):
        self._le = skl.preprocessing.LabelEncoder().fit(classes)
        self.classes_ = list(self._le.classes_)
        self.n_classes_ = len(self.classes_)

        return self

    def fit(self, X, y):
        self.set_classes(y)
        xgb_options = self.get_xgb_params()
        xgb_options['objective'] = 'binary:logistic'

        return self._run_booster(X, y, xgb_options)

    def _run_booster(self, X, y, xgb_options):
        evals_result = {}
        eval_metric = getattr(self, 'eval_metric', None)
        eval_set = getattr(self, 'eval_set', None)
        early_stopping_rounds = getattr(self, 'early_stopping_rounds', None)

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

        training_labels = self._transform_y(y)
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


class MultiClassXGB(LogisticXGB):
    def _transform_y(self, y):
       return self._le.transform(y)

    def fit(self, X, y, classes=None):
        if not hasattr(self, 'classes_'):
            search_classes = classes if classes is not None else y

            if hasattr(search_classes, 'cat'):
                self.set_classes(search_classes.cat.categories)
            else:
                self.set_classes(search_classes)

        xgb_options = self.get_xgb_params()
        xgb_options["objective"] = "multi:softprob"
        xgb_options['num_class'] = self.n_classes_

        return self._run_booster(X, y, xgb_options)

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


class TuneXGB:
    def __init__(self, model_class, X, y, metric=None, learning_rate=0.2, random_state=1234, early_stopping_rounds=25,
                cv=None, train_size=0.7, n_iter=1, max_delta_step=1):

        self.model_class = model_class
        if metric is None:
            metric = 'auc' if model_class == LogisticXGB else 'mlogloss'

        self.metric = metric
        self.metric_ascending = (metric != 'auc')

        self.learning_rate = learning_rate
        self.max_delta_step = max_delta_step
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        self.X = X
        self.y = y
        self.cv = cv if cv is not None else skl.cross_validation.ShuffleSplit(
            len(y), train_size=train_size, n_iter=n_iter, random_state=random_state
        )

        self.best_max_depth = None
        self.best_min_child_weight = None
        self.best_gamma = None
        self.best_subsample = None
        self.best_colsample_bytree = None

    def get_model_params(self, params=None):
        params = params.copy() if params is not None else {}

        if 'n_estimators' not in params:
            params['n_estimators'] = 3000

        def add_param(param, self_param=None):
            if self_param is None:
                self_param = param

            if param not in params:
                self_param_value = getattr(self, self_param, None)
                if self_param_value is not None:
                    params[param] = self_param_value

        add_param('learning_rate')
        add_param('max_delta_step')
        add_param('max_depth', 'best_max_depth')
        add_param('min_child_weight', 'best_min_child_weight')
        add_param('gamma', 'best_gamma')
        add_param('subsample', 'best_subsample')
        add_param('colsample_bytree', 'best_colsample_bytree')
        add_param('seed', 'random_state')

        return params

    def build_model(self, params=None):
        params = self.get_model_params(params)
        return self.model_class(**params)

    def apply_cv(self, model, cv_num=0):
        train, eval = list(self.cv)[cv_num]
        self.model_apply_cv_eval(model, eval)

    def model_apply_cv_eval(self, model, ix):
        X_eval = self.X.iloc[ix, :]
        y_eval = self.y.iloc[ix]
        model.set_eval(X_eval, y_eval, self.metric, early_stopping_rounds=self.early_stopping_rounds)

    def train_models_cv(self, params=None, ntrees=None):
        num = 0
        for train, eval in self.cv:
            X_train = self.X.iloc[train, :]
            y_train = self.y.iloc[train]

            if ntrees is not None:
                params = params.copy() if params is not None else {}
                params['n_estimators'] = ntrees[num]

            model = self.build_model(params)
            self.model_apply_cv_eval(model, eval)
            model.fit(X_train, y_train)
            num += 1

            yield model

    def train_models_cv_score(self, params=None, mean=True, progress=None, model_ntrees=None):
        scores = []
        ntree = []
        models = []

        if progress == True:
            progress = FloatProgress(min=0, max=len(self.cv))
            display(progress)

        for model in self.train_models_cv(params, model_ntrees):
            scores.append(model.booster().best_score)
            ntree.append(model.booster().best_ntree_limit)

            if not mean:
                models.append(model)

            if progress is not None:
                progress.value += 1

        if params is None:
            params = {}

        if mean:
            result = params.copy()
            result['score'] = np.mean(scores)
            result['ntree'] = np.mean(ntree)
        else:
            result = pd.DataFrame({
                'score': scores,
                'ntree': ntree,
                'model': models,
            }).sort_values('score', ascending=self.metric_ascending)

        return result

    def score_params(self, param_ranges):
        scores = pd.DataFrame(columns=['score', 'ntree'] + param_ranges.keys())
        product = list(itertools.product(*param_ranges.values()))
        progress = FloatProgress(min=0, max=len(product)*len(self.cv))
        display(progress)

        for param_values in product:
            params = dict(zip(param_ranges.keys(), param_values))
            sys.stderr.write(str(params) + '\n')
            scores = scores.append(self.train_models_cv_score(params, progress=progress), ignore_index=True)

        return scores.sort_values('score', ascending=self.metric_ascending)

    def detail_range(self, center, delta, min=None, max=None):
        drange = []
        left, right = center - delta, center + delta

        if (min is None) or left >= min:
            drange.append(left)

        drange.append(center)

        if (max is None) or right <= max:
            drange.append(right)

        return drange

    def score_depth(self, max_depth_range=None, min_child_weight_range=None):
        return self.score_params({
            'max_depth': max_depth_range if max_depth_range is not None else range(3, 10, 2),
            'min_child_weight': min_child_weight_range if min_child_weight_range is not None else range(1, 6, 2),
        })

    def tune_depth(self):
        scores = self.score_depth()
        best_score = scores.iloc[0]
        self.best_max_depth = int(best_score['max_depth'])
        self.best_min_child_weight = int(best_score['min_child_weight'])

        max_depth_range = self.detail_range(self.best_max_depth, 1, min=1)
        min_child_weight_range = self.detail_range(self.best_min_child_weight, 1, min=1)
        scores2 = self.score_depth(max_depth_range, min_child_weight_range)
        best_score2 = scores2.iloc[0]
        self.best_max_depth = int(best_score2['max_depth'])
        self.best_min_child_weight = int(best_score2['min_child_weight'])

        scores = pd.concat([scores, scores2], ignore_index=True)
        return scores.drop_duplicates().sort_values('score', ascending=self.metric_ascending)

    def score_gamma(self, gamma_range=None):
        return self.score_params({
            'gamma': gamma_range if gamma_range is not None else np.arange(0.0, 0.5, 0.1),
        })

    def tune_gamma(self):
        scores = self.score_gamma()
        best_score = scores.iloc[0]
        self.best_gamma = best_score['gamma']

        gamma_range = self.detail_range(self.best_gamma, 0.05, min=0)
        scores2 = self.score_gamma(gamma_range)
        best_score2 = scores2.iloc[0]
        self.best_gamma = best_score2['gamma']

        scores = pd.concat([scores, scores2], ignore_index=True)
        return scores.drop_duplicates().sort_values('score', ascending=self.metric_ascending)

    def score_samples(self, subsample_range=None, colsample_bytree_range=None):
        return self.score_params({
            'subsample': subsample_range if subsample_range is not None else np.arange(1.0, 0.5, -0.1),
            'colsample_bytree': colsample_bytree_range if colsample_bytree_range is not None else np.arange(1.0, 0.5, -0.1),
        })

    def tune_sample(self):
        scores = self.score_samples()
        best_score = scores.iloc[0]
        self.best_subsample = best_score['subsample']
        self.best_colsample_bytree = best_score['colsample_bytree']

        subsample_range = self.detail_range(self.best_subsample, 0.05, max=1)
        colsample_bytree_range = self.detail_range(self.best_colsample_bytree, 0.05, max=1)
        scores2 = self.score_samples(subsample_range, colsample_bytree_range)
        best_score2 = scores2.iloc[0]
        self.best_subsample = best_score2['subsample']
        self.best_colsample_bytree = best_score2['colsample_bytree']

        scores = pd.concat([scores, scores2], ignore_index=True)
        return scores.drop_duplicates().sort_values('score', ascending=self.metric_ascending)

    def get_cv_train_size(self):
        for train, test in self.cv:
            return len(train)
