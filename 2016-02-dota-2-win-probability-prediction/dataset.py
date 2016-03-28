from helpers.data import *
from helpers.model import LogisticXGB, model_train_cv_parallel
from synergy import get_synergy, get_synergy_aggregate, synergy_calc_cv_parallel


def select_players(teams=None, nums=None, features=None):
    if teams is None:
        teams = ['r', 'd']

    if nums is None:
        nums = [1, 2, 3, 4, 5]

    if features is None:
        features = ['hero', 'level', 'xp', 'gold', 'lh', 'kills', 'deaths', 'items']

    columns = []
    for team in teams:
        for num in nums:
            for feature in features:
                columns.append(team + str(num) + '_' + feature)

    return columns


def data_v1(data):
    for df in data.dfs().itervalues():
        df.index = df['match_id']
        df.drop(['match_id'], axis=1, inplace=True)

    data.dset('train')['y'] = data.get('train')['radiant_win']

    sel = select_players()
    cols = ['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire']
    for dset, df in data.dfs().iteritems():
        data.dset(dset)['players'] = df[sel]
        df.drop(sel, axis=1, inplace=True)
        for col in cols:
            if col in df:
                df.drop([col], axis=1, inplace=True)


def data_v2(data):
    for dset, df in data.dfs().iteritems():
        df['lobby_type'] = df['lobby_type'].replace({
            0: 'public',
            1: 'practice',
            7: 'ranked',
        }).astype('category')
        data.dset(dset)['data'] = pd.get_dummies(df, columns=['lobby_type'])


def data_v3(data):
    for dset, df in data.dfs().iteritems():
        df['first_blood_team'] = df['first_blood_team'].replace({
            1: -1,
            0: 1
        }).fillna(0)

        df['first_blood_time'] += 91
        df['first_blood_time'] = df['first_blood_time'].fillna(0)

        df['first_blood_player1'] += 1
        df['first_blood_player2'] += 1
        df['first_blood_player1'] = df['first_blood_player1'].fillna(0)
        df['first_blood_player2'] = df['first_blood_player2'].fillna(0)

        fb_cols = ['first_blood_time', 'first_blood_team', 'first_blood_player1', 'first_blood_player2']
        data.dset(dset)['fb'] = df[fb_cols]
        df.drop(fb_cols, axis=1, inplace=True)


def data_v4(data):
    item_times = ['bottle_time', 'courier_time', 'flying_courier_time', 'first_ward_time']
    teams = {
        'r': 'radiant',
        'd': 'dire',
    }

    for dset, df in data.dfs().iteritems():
        cols = []
        for t, team in teams.items():
            for time_col in item_times:
                col1 = team + '_' + time_col
                col2 = t + '_' + time_col
                cols.append(col2)

                df[col1] += 91
                df[col2] = df[col1].fillna(0)
                df.drop([col1], axis=1, inplace=True)

        data.dset(dset)['teams'] = df[cols]
        df.drop(cols, axis=1, inplace=True)


def data_v5(data):
    item_cnts = ['tpscroll_count', 'boots_count', 'ward_observer_count', 'ward_sentry_count']
    teams = {
        'r': 'radiant',
        'd': 'dire',
    }

    for dset, df in data.dfs().iteritems():
        for t, team in teams.items():
            for cnt_col in item_cnts:
                col1 = team + '_' + cnt_col
                col2 = t + '_' + cnt_col
                data.get(dset, df='teams')[col2] = df[col1]
                df.drop([col1], axis=1, inplace=True)


def data_v6(data):
    for dset, df in data.dfs().iteritems():
        data.dset(dset)['match'] = df
        del data.dsets[dset]['data']


def data_v7(data):
    heroes = pd.read_csv('./data/dictionaries/heroes.csv')
    for dset, df in data.dfs('players').iteritems():
        pick = pd.DataFrame(0, index=df.index, columns=heroes['id'])
        for match_id in df.index:
            for p in range(1, 6):
                pick.ix[match_id, df.ix[match_id, 'r%d_hero' % p]] = 1
                pick.ix[match_id, df.ix[match_id, 'd%d_hero' % p]] = -1

        pick.columns = heroes['name']
        data.dset(dset)['heroes'] = pick


def data_v8(data):
    for dset, df in data.dfs('players').iteritems():
        df2 = data.get(dset, df='teams')
        for team in ['r', 'd']:
            for feature in ['level', 'xp', 'gold', 'lh', 'kills', 'deaths', 'items']:
                df2[team + '_' + feature] = df[select_players([team], features=[feature])].sum(axis=1)


def data_v9(data):
    for df in data.dfs('teams').itervalues():
        for feature in ['level', 'xp', 'gold', 'lh', 'items', 'kills']:
            df['r_' + feature + '_ratio'] = ((df['r_' + feature] - df['d_' + feature]) / (df['r_' + feature] + df['d_' + feature])).fillna(0)


def data_v10(data):
    for dset, df in data.dfs('teams').iteritems():
        features = ['gold', 'xp', 'lh', 'items']
        diffs = pd.DataFrame(index=df.index)
        for feature in features:
            diffs[feature + '_diff'] = df['r_' + feature] - df['d_' + feature]

        diffs['kdr'] = df['r_kills'] / (df['d_kills'] + 0.0001)
        data.dset(dset)['team_diffs'] = diffs


def data_v11(data):
    for dset, teams in data.dfs('teams').iteritems():
        X = data.dset(dset)['team_diffs']
        X['bottle'] = (teams['r_bottle_time'] > 0) * 1 + (teams['d_bottle_time'] > 0) * -1
        X['courier'] = (teams['r_courier_time'] == 0) * -1 + (teams['d_courier_time'] == 0) * 1
        X['flying_courier'] = (teams['r_flying_courier_time'] > 0) * 1 + (teams['d_flying_courier_time'] > 0) * -1
        X['first_ward'] = ((teams['r_first_ward_time'] > 0) & (teams['r_first_ward_time'] < 150)) * 1 + \
                          ((teams['d_first_ward_time'] > 0) & (teams['d_first_ward_time'] < 150)) * -1

        scrolls = (teams['d_tpscroll_count'] - teams['r_tpscroll_count'])
        scrolls[scrolls > 10] = 10
        scrolls[scrolls < -10] = -10
        X['tpscroll_count'] = scrolls * 0.1

        boots = teams['r_boots_count'] - teams['d_boots_count']
        boots[boots > 10] = 10
        boots[boots < -10] = -10
        X['boots_count'] = boots * 0.1


def data_v12(data):
    desc = data.get('train', 'heroes').describe().T
    drop_heroes = list(desc[desc['std'] == 0].index)
    for df in data.dfs('heroes').itervalues():
        df.drop(drop_heroes, axis=1, inplace=True)


def data_v13(data):
    heroes = data.get('train', 'heroes')
    y = data.get('train', 'y')
    cv = skl.cross_validation.KFold(len(y), n_folds=16, shuffle=True, random_state=1234)
    data.dset('train')['synergy'] = synergy_calc_cv_parallel(heroes, y, cv)


def data_v14(data):
    stats = pd.read_csv('data/dictionaries/hero_stats.csv', index_col=0)
    stat_cols = ['Carry', 'Disabler', 'Lane_support', 'Initiator', 'Jungler', 'Support', 'Durable', 'Pusher', 'Nuker',
                 'Escape', 'Melee', 'Ranged', 'Strength', 'Intelligence', 'Agility']

    for dset, players in data.dfs('players').iteritems():
        X = pd.DataFrame(0, index=players.index, columns=[t + '_' + s for t, s in itertools.product(['r', 'd'], stat_cols)])
        for match_id in X.index:
            match = players.ix[match_id, :]
            for team in ['r', 'd']:
                for stat in stat_cols:
                    for h in match[select_players(team, features=['hero'])]:
                        X.ix[match_id, team + '_' + stat] += stats.ix[h, stat]

        data.dset(dset)['hero_roles'] = X


def data_v15(data):
    heroes = data.get('train', 'heroes')
    y = data.get('train', 'y')
    synergy, anti_synergy = get_synergy(heroes, y, 10)
    data.dset('train')['synergy_matrix'] = synergy
    data.dset('train')['anti_synergy_matrix'] = anti_synergy


def data_v16(data):
    heroes = data.get('test', 'heroes')
    synergy = data.get('train', 'synergy_matrix')
    anti_synergy = data.get('train', 'anti_synergy_matrix')
    data.dset('test')['synergy'] = get_synergy_aggregate(heroes, synergy, anti_synergy)


def data_v17(data):
    for dset, df in data.dfs('synergy').iteritems():
        X = pd.DataFrame(index=df.index)
        X['synergy'] = df['r_synergy_mean'] - df['d_synergy_mean']
        X['anti_synergy'] = df['r_anti_synergy_mean']
        data.dset(dset)['synergy_sum'] = X


def data_v18(data):
    items = pd.read_csv('data/items.csv', index_col=0)
    test_items = pd.read_csv('data/items_test.csv', index_col=0)

    items_columns = [col[2:] for col in items.columns][:254]
    items_diff = pd.DataFrame(0, index=items.index, columns=items_columns)
    test_items_diff = pd.DataFrame(0, index=test_items.index, columns=items_columns)

    for col in items_columns:
        items_diff[col] = items['r_' + col] - items['d_' + col]
        test_items_diff[col] = test_items['r_' + col] - test_items['d_' + col]

    desc = items_diff.describe().T
    empty_items = list(desc[desc['std'] == 0].index)

    data.dset('train')['items_diff'] = items_diff.drop(empty_items, axis=1)
    data.dset('test')['items_diff'] = test_items_diff.drop(empty_items, axis=1)


def data_v19(data):
    abilities = pd.read_csv('data/abilities.csv', index_col=0)
    test_abilities = pd.read_csv('data/abilities_test.csv', index_col=0)

    abilities_columns = [col[2:] for col in abilities.columns][:568]
    abilities_diff = pd.DataFrame(0, index=abilities.index, columns=abilities_columns)
    test_abilities_diff = pd.DataFrame(0, index=test_abilities.index, columns=abilities_columns)

    for col in abilities_columns:
        abilities_diff[col] = abilities['r_' + col] - abilities['d_' + col]
        test_abilities_diff[col] = test_abilities['r_' + col] - test_abilities['d_' + col]

    desc = abilities_diff.describe().T
    empty_abilities = list(desc[desc['std'] == 0].index)

    data.dset('train')['abilities_diff'] = abilities_diff.drop(empty_abilities, axis=1)
    data.dset('test')['abilities_diff'] = test_abilities_diff.drop(empty_abilities, axis=1)


def data_v20(data):
    def get_bb(bb):
        bb['bb_count'] = bb['d_bb_count'] - bb['r_bb_count']
        return bb['bb_count']

    data.dset('train')['bb'] = get_bb(pd.read_csv('data/bb.csv', index_col=0))
    data.dset('test')['bb'] = get_bb(pd.read_csv('data/bb_test.csv', index_col=0))


def data_v21(data):
    def acc_diff(row, feature):
        r = np.sort(row[select_players(['r'], features=[feature])].values)
        d = np.sort(row[select_players(['d'], features=[feature])].values)

        return pd.Series({'acc_' + feature + '_' + str(abs(n)): r[n:].sum() - d[n:].sum() for n in range(-1, -6, -1)})

    for dset, df in data.dfs('players').iteritems():
        accs = []
        for feature in ['gold', 'xp', 'lh', 'kills']:
            accs.append(df.apply(lambda row: acc_diff(row, feature), axis=1))

        data.dset(dset)['team_accs'] = pd.concat(accs, axis=1)


def data_v22(data):
    def get_roles_bag(roles):
        role_cols = ['Carry', 'Disabler', 'Lane_support', 'Initiator', 'Jungler', 'Support', 'Durable', 'Pusher', 'Nuker', 'Escape']
        type_cols = ['Melee', 'Ranged']
        attr_cols = ['Strength', 'Intelligence', 'Agility']

        bag = pd.DataFrame(index=roles.index)
        for group in [role_cols, type_cols, attr_cols]:
            for f1, f2 in itertools.combinations_with_replacement(group, 2):
                r, d = roles['r_' + f1], roles['d_' + f2]
                bag[f1 + '_x_' + f2] = (r > d) * 1 + (d > r) * -1

        return bag

    for dset, roles in data.dfs('hero_roles').iteritems():
        data.dset(dset)['hero_roles_bag'] = get_roles_bag(roles)


def data_v23(data):
    model = LogisticXGB(n_estimators=2, max_depth=6, learning_rate=0.001, nthread=1)
    X = data.get('train', 'match')[['start_time']]
    y = data.get('train', 'y')

    train_models = pd.DataFrame(index=X.index)
    train_models['time_xgb'] = model_train_cv_parallel(model, X, y)['predict'] - 0.5
    data.dset('train')['models'] = train_models

    X_test = data.get('test', 'match')[['start_time']]
    test_models = pd.DataFrame(index=X_test.index)
    model.fit(X, y)
    test_models['time_xgb'] = model.predict_proba(X_test)[:, 1] - 0.5
    data.dset('test')['models'] = test_models


def data_v24(data):
    scaler = skl.preprocessing.StandardScaler()

    def get_y():
        return data.get('train', df='y')

    def get_X(dset='train'):
        return data.extract(dset, [
            ('models', ['time_xgb']),
            ('synergy_sum', None),
            ('hero_roles_bag', None),
            ('match', ['lobby_type_practice', 'lobby_type_public', 'lobby_type_ranked'])
        ], scaler=scaler)

    model = skl.linear_model.LogisticRegression(random_state=123, C=0.001)
    X, y = get_X(), get_y()

    data.get('train', 'models')['pre_match_linear'] = model_train_cv_parallel(model, X, y)['predict'] - 0.5

    model.fit(X, y)
    data.get('test', 'models')['pre_match_linear'] = model.predict_proba(get_X('test'))[:, 1] - 0.5


def data_v25(data):
    def get_y():
        return data.get('train', df='y')

    def get_X(dset='train'):
        return data.extract(dset, [
            ('models', ['time_xgb']),
            ('synergy_sum', None),
            ('hero_roles_bag', None),
            ('match', ['lobby_type_practice', 'lobby_type_public', 'lobby_type_ranked'])
        ])

    model = LogisticXGB(n_estimators=100, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, seed=1234, nthread=1)
    X, y = get_X(), get_y()

    data.get('train', 'models')['pre_match_xgb'] = model_train_cv_parallel(model, X, y)['predict'] - 0.5

    model.fit(X, y)
    data.get('test', 'models')['pre_match_xgb'] = model.predict_proba(get_X('test'))[:, 1] - 0.5


def data_v26(data):
    for df_sparse in ['items_diff', 'abilities_diff', 'heroes']:
        for dset, df in data.dfs(df_sparse).iteritems():
            if type(df) != pd.sparse.frame.SparseDataFrame:
                data.dset(dset)[df_sparse] = df.to_sparse(0)


def data_v27(data):
    def count_items(items, columns=None):
        if columns is None:
            item_counts = pd.DataFrame(index=items.index)
        else:
            item_counts = pd.DataFrame(0, index=items.index, columns=columns)

        for col in items:
            n = 1
            while True:
                col_name = col + '_' + str(n)
                if columns is not None:
                    if col_name not in columns:
                        break

                counts = items[col][(items[col] >= n) | (items[col] <= -n)]

                if (n > 1) and (len(counts) < 100):
                    break

                if len(counts) > 0:
                    item_counts[col_name] = (counts > 0) * 1 + (counts < 0) * -1

                n += 1

        return item_counts.fillna(0).to_sparse(0)

    train = data.dset('train')['item_counts'] = count_items(data.get('train', 'items_diff'))
    data.dset('test')['item_counts'] = count_items(data.get('test', 'items_diff'), train.columns)


def data_v28(data):
    def count_abilities(abilities, columns=None):
        if columns is None:
            ability_counts = pd.DataFrame(index=abilities.index)
        else:
            ability_counts = pd.DataFrame(0, index=abilities.index, columns=columns)

        for col in abilities:
            n = 1
            while True:
                col_name = col + '_' + str(n)
                if columns is not None:
                    if col_name not in columns:
                        break

                counts = abilities[col][(abilities[col] >= n) | (abilities[col] <= -n)]

                if (n > 1) and (len(counts) < 10):
                    break

                if len(counts) > 0:
                    ability_counts[col_name] = (counts > 0) * 1 + (counts < 0) * -1

                n += 1

        return ability_counts.fillna(0).to_sparse(0)

    train = data.dset('train')['ability_counts'] = count_abilities(data.get('train', 'abilities_diff'))
    data.dset('test')['ability_counts'] = count_abilities(data.get('test', 'abilities_diff'), train.columns)


def data_v29(data):
    def count_player_stats(team_accs, stat, stat_step, columns=None, min_games=100):
        if columns is None:
            stat_counts = pd.DataFrame(index=team_accs.index)
        else:
            stat_counts = pd.DataFrame(0, index=team_accs.index, columns=columns)

        for k in range(1, 6):
            n = 1
            while True:
                limit = n * stat_step * k
                col_name = stat + '_' + str(k) + '_' + str(limit)
                if columns is not None:
                    if col_name not in columns:
                        break

                col = 'acc_' + stat + '_' + str(k)
                counts = team_accs[col][(team_accs[col] >= limit) | (team_accs[col] <= -limit)]

                if (n > 1) and (len(counts) < min_games):
                    break

                if len(counts) > 0:
                    stat_counts[col_name] = (counts > 0) * 1 + (counts < 0) * -1

                n += 1

        return stat_counts.fillna(0).to_sparse(0)

    for stat, stat_step in [('gold', 100), ('xp', 100), ('lh', 4), ('kills', 1)]:
        train = data.dset('train')[stat + '_counts'] = count_player_stats(data.get('train', 'team_accs'), stat, stat_step)
        data.dset('test')[stat + '_counts'] = count_player_stats(data.get('test', 'team_accs'), stat, stat_step, train.columns)


def data_v30(data):
    scaler = skl.preprocessing.StandardScaler()

    def get_y():
        return data.get('train', df='y')

    def get_X(dset='train'):
        X = data.extract(dset, [
            ('match', ['lobby_type_practice']),
            ('models', ['time_xgb']),
            ('synergy_sum', None),
            ('item_counts', None),
            ('ability_counts', None),
            ('team_diffs', ['first_ward']),
            ('fb', ['first_blood_team']),
            ('gold_counts', None),
            ('xp_counts', None),
            ('lh_counts', None),
            ('kills_counts', None),
        ]).to_sparse(0)

        scale = ['time_xgb', 'anti_synergy', 'synergy']
        X[scale] = scaler.fit_transform(X[scale]) if dset == 'train' else scaler.transform(X[scale])

        return X

    model = skl.linear_model.LogisticRegression(random_state=1234, C=0.005)
    X, y = get_X(), get_y()

    data.get('train', 'models')['common_linear'] = model_train_cv_parallel(model, X, y, n_jobs=1)['predict'] - 0.5

    model.fit(X, y)
    data.get('test', 'models')['common_linear'] = model.predict_proba(get_X('test'))[:, 1] - 0.5


def data_v31(data):
    def get_y():
        return data.get('train', df='y')

    def get_X(dset='train'):
        return data.extract(dset, [
            ('match', ['lobby_type_practice']),
            ('models', ['time_xgb']),
            ('synergy_sum', None),
            ('item_counts', None),
            ('ability_counts', None),
            ('team_diffs', ['first_ward']),
            ('fb', ['first_blood_team']),
            ('gold_counts', None),
            ('xp_counts', None),
            ('lh_counts', None),
            ('kills_counts', None),
            ('hero_roles_bag', None),
        ]).to_sparse(0)

    model = LogisticXGB(n_estimators=500, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.6, max_delta_step=1, seed=1234)
    X, y = get_X(), get_y()

    data.get('train', 'models')['common_xgb'] = model_train_cv_parallel(model, X, y, n_jobs=1)['predict'] - 0.5

    model.fit(X, y)
    data.get('test', 'models')['common_xgb'] = model.predict_proba(get_X('test'))[:, 1] - 0.5


def data_v32(data):
    def get_y():
        return data.get('train', df='y')

    def get_X(dset='train'):
        return data.extract(dset, [
            ('match', ['lobby_type_practice']),
            ('models', ['time_xgb', 'pre_match_linear', 'pre_match_xgb', 'common_linear', 'common_xgb']),
        ]).to_sparse(0)

    model = LogisticXGB(n_estimators=300, learning_rate=0.01, max_depth=4, max_delta_step=1, seed=1234)
    X, y = get_X(), get_y()

    data.get('train', 'models')['ensemble1_xgb'] = model_train_cv_parallel(model, X, y, n_jobs=1)['predict'] - 0.5

    model.fit(X, y)
    data.get('test', 'models')['ensemble1_xgb'] = model.predict_proba(get_X('test'))[:, 1] - 0.5

def data_v33(data):
    def get_edges_sample():
        y = data.get('train', 'y')
        models = data.get('train', 'models')
        predict = models['ensemble1_xgb'] + 0.5

        results = pd.DataFrame({'y': y, 'predict': predict})
        fn = results[(results['predict'] < 0.15) & (results['y'] == 1)].sample(250, random_state=42)
        tn = results[(results['predict'] < 0.15) & (results['y'] == 0)].sample(250, random_state=42)
        fp = results[(results['predict'] > 0.85) & (results['y'] == 0)].sample(250, random_state=42)
        tp = results[(results['predict'] > 0.85) & (results['y'] == 1)].sample(250, random_state=42)

        return pd.concat([fn, tn, fp, tp])

    sample = get_edges_sample()
    scaler = skl.preprocessing.StandardScaler()
    ix = sample.index

    def get_y(sample=True):
        if sample:
            return data.get('train', df='y').ix[ix]
        else:
            return data.get('train', df='y')

    def get_X(dset='train', sample=True):
        i = ix if sample else None
        X = data.extract(dset, [
            ('match', ['lobby_type_practice']),
            ('team_diffs', None),
        ], ix=i, scaler=scaler)

        return X

    model = skl.linear_model.LogisticRegression(random_state=1234, C=0.001)
    X, y = get_X(), get_y()
    model.fit(X, y)

    data.get('train', 'models')['simple_linear'] = model.predict_proba(get_X('train', sample=False))[:, 1] - 0.5
    data.get('test', 'models')['simple_linear'] = model.predict_proba(get_X('test', sample=False))[:, 1] - 0.5


data = DataProvider({
    'train': 'data/features.csv',
    'test': 'data/features_test.csv'
}, [data_v1, data_v2, data_v3, data_v4, data_v5, data_v6, data_v7, data_v8, data_v9, data_v10, data_v11, data_v12,
    data_v13, data_v14, data_v15, data_v16, data_v17, data_v18, data_v19, data_v20, data_v21, data_v22, data_v23,
    data_v24, data_v25, data_v26, data_v27, data_v28, data_v29, data_v30, data_v31, data_v32, data_v33])
