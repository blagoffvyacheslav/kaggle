from helpers.data import itertools, pd, np, Parallel, delayed


def get_hero_matrix(heroes):
    return pd.DataFrame(0, index=heroes.columns, columns=heroes.columns)


def add_synergy_combinations(h_combs, synergy_matrix, symmetric=True):
    for h1, h2 in h_combs:
        synergy_matrix.ix[h1, h2] += 1
        if symmetric:
            synergy_matrix.ix[h2, h1] += 1


def get_match_comb(match_id, heroes, y):
    row = heroes.ix[match_id, :]
    win = y.ix[match_id]

    r_heroes, d_heroes = list(row[row > 0].index), list(row[row < 0].index)
    win_heroes = r_heroes if win == 1 else d_heroes
    lose_heroes = r_heroes if win == 0 else d_heroes

    win_comb = list(itertools.combinations(win_heroes, 2))
    lose_comb = list(itertools.combinations(lose_heroes, 2))
    win_prod = list(itertools.product(win_heroes, lose_heroes))

    return win_comb, lose_comb, win_prod


def normalize_synergy(synergy, matches, min_matches):
    return (((synergy / matches[matches > min_matches]) - 0.5) * 2).replace([np.inf, -np.inf], np.nan).fillna(0)


def get_synergy(heroes, y, min_matches=0):
    matches = get_hero_matrix(heroes)
    synergy = get_hero_matrix(heroes)

    anti_matches = get_hero_matrix(heroes)
    anti_synergy = get_hero_matrix(heroes)

    for match_id in heroes.index:
        win_comb, lose_comb, win_prod = get_match_comb(match_id, heroes, y)

        add_synergy_combinations(win_comb, matches)
        add_synergy_combinations(lose_comb, matches)
        add_synergy_combinations(win_comb, synergy)

        add_synergy_combinations(win_prod, anti_matches)
        add_synergy_combinations(win_prod, anti_synergy, False)

    return normalize_synergy(synergy, matches, min_matches), normalize_synergy(anti_synergy, anti_matches, min_matches)


def get_synergy_aggregate(heroes, synergy, anti_synergy):
    aggr = pd.DataFrame(index=heroes.index)
    for match_id in heroes.index:
        row = heroes.ix[match_id, :]
        r_heroes, d_heroes = list(row[row > 0].index), list(row[row < 0].index)

        for t, h in {'r': r_heroes, 'd': d_heroes}.items():
            team_synergy = []
            for h1, h2 in itertools.combinations(h, 2):
                team_synergy.append(synergy.ix[h1, h2])

            aggr.ix[match_id, t + '_synergy_min'] = np.min(team_synergy)
            aggr.ix[match_id, t + '_synergy_max'] = np.max(team_synergy)
            aggr.ix[match_id, t + '_synergy_sum'] = np.sum(team_synergy)
            aggr.ix[match_id, t + '_synergy_mean'] = np.mean(team_synergy)
            aggr.ix[match_id, t + '_synergy_std'] = np.std(team_synergy)

        r_anti_synergy = []
        for h1, h2 in itertools.product(r_heroes, d_heroes):
            r_anti_synergy.append(anti_synergy.ix[h1, h2])

        aggr.ix[match_id, 'r_anti_synergy_min'] = np.min(r_anti_synergy)
        aggr.ix[match_id, 'r_anti_synergy_max'] = np.max(r_anti_synergy)
        aggr.ix[match_id, 'r_anti_synergy_sum'] = np.sum(r_anti_synergy)
        aggr.ix[match_id, 'r_anti_synergy_mean'] = np.mean(r_anti_synergy)
        aggr.ix[match_id, 'r_anti_synergy_std'] = np.std(r_anti_synergy)

    return aggr


def synergy_calc_cv(fold, heroes, y):
    train, test = fold
    synergy, anti_synergy = get_synergy(heroes.iloc[train, :], y.iloc[train], 10)
    return get_synergy_aggregate(heroes.iloc[test, :], synergy, anti_synergy)


def synergy_calc_cv_parallel(heroes, y, cv):
    return pd.concat(Parallel(n_jobs=8, verbose=True)(delayed(synergy_calc_cv)(fold, heroes, y) for fold in cv))
