# ToDo: On dfsp rescale each timestamp using dfmp and dfdp
# ToDo: Do better imputation, possibly by adding previous timestamp data for factor (maybe use fancyimpute)
# ToDo: Add flight/fight indicator to prediction - Weight stocks -1 to 1 by predicted volatility to create new factors (and y), and then predict the weighted y
# ToDo: Look into using moving averages of the factors
# ToDo: Use full dataset
# ToDo: Use other models in an ensemble approach
# ToDo: Reduce the number of factors used
# ToDo: Add polynomials
# ToDo: In final submission, submit two models, one with training mean market return, and one 0.00008 higher.

def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight,
                  multioutput=multioutput)
    r = (np.sign(r2) * np.sqrt(np.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r


def createPredictionModel(name, model, data, factors, outcome):
    print("\n" + name)

    cvFolds = GroupKFold(n_splits=5)

    filteredData = data[~data[outcome].isnull()]

    params = model.get_params()
    params["n_jobs"] = (1, -1)[testing]
    params["cv"] = cvFolds.split(filteredData, groups=filteredData.index.get_level_values(0))
    model.set_params(**params)

    # Fit the model:
    model.fit(filteredData[factors], filteredData[outcome])

    # Coefs
    coefs = pd.DataFrame(model.coef_, index=factors, columns=["coef"])
    coefs = coefs.loc[coefs.coef != 0]
    if coefs.size > 0:
        coefs = coefs.assign(absCoef=abs(coefs.coef))
        coefs.sort_values(by="absCoef", na_position="first", inplace=True)
        coefs.drop("absCoef", axis=1, inplace=True)
        print(coefs)

        # Make predictions on training set
        predictions = model.predict(filteredData[factors])
        cor = stats.pearsonr(predictions, filteredData[outcome])
        print("Training Correlation: %s" % "{0:.3%}".format(cor[0]))
        rScore = r_score(filteredData[outcome], predictions)
        print("Training R Score: %s" % "{0:.3%}".format(rScore))
    else:
        print("Model Rejected - No Coefficients")

    if str(model.__class__) == "<class 'sklearn.linear_model.coordinate_descent.ElasticNetCV'>":
        print("L1 Ratio: ", model.l1_ratio_)
        print("Alpha: ", model.alpha_)

    if testing:
        # Perform CV
        cvmodel = lm.ElasticNetCV(l1_ratio=model.l1_ratio_, alphas=[model.alpha_], n_jobs=1)  # (1, -1)[testing])
        cvFolds = GroupKFold(n_splits=5)
        cvPredictions = ms.cross_val_predict(cvmodel, filteredData[factors], filteredData[outcome],
                                             cv=cvFolds.split(filteredData,
                                                              groups=filteredData.index.get_level_values(0)),
                                             n_jobs=(1, -1)[testing])

        # Make predictions to cv set
        cor = stats.pearsonr(cvPredictions, filteredData[outcome])
        print("C-V Correlation: %s" % "{0:.3%}".format(cor[0]))
        rScore = r_score(filteredData[outcome], cvPredictions)
        print("C-V R Score: %s" % "{0:.3%}".format(rScore))
        model.cvPredictions = cvPredictions

    model.coefs = coefs

    return model


if __name__ == '__main__':
    testing = True
    full = True

    import kagglegym
    import numpy as np
    import pandas as pd
    import sklearn as sl
    import scipy.stats as stats
    from timeit import default_timer as timer
    import math as math
    from sklearn import linear_model as lm
    from sklearn import model_selection as ms
    from sklearn.model_selection import GroupKFold
    import random
    from sklearn.metrics import r2_score
    import gc

    # The "environment" is our interface for code competitions
    env = kagglegym.make()

    # We get our initial observation by calling "reset"
    observation = env.reset()
    print("Train has {} rows".format(len(observation.train)))
    print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

    # Load data frame
    df = observation.train
    df.set_index(["timestamp", "id"], inplace=True)
    factors = [factor for factor in df.columns.values if not factor in ["id", "timestamp", "y"]]

    dfs = df

    # Create small data frame
    random.seed()
    if ~full & testing:
        smallSize = 200000
        timestamps = list(dfs.index.get_level_values(0).unique().values)
        timestamps = [timestamps[i] for i in sorted(random.sample(range(len(timestamps)),
                                                                  round(math.sqrt(smallSize / dfs.shape[0]) * len(
                                                                      timestamps))))]
        dfs = dfs.loc[timestamps]
        ids = list(dfs.index.get_level_values(1).unique().values)
        ids = [ids[i] for i in sorted(random.sample(range(len(ids)), round(smallSize / dfs.shape[0] * len(ids))))]
        dfs = dfs.loc[dfs.index.get_level_values(1).isin(ids)]

    # PreProcess data (normalise to percentile)
    dfsp = 99 * dfs[factors].rank(method="average", na_option="keep", pct=True) // 1 / 100.0

    # create factor range z-score lookup
    factorRange = pd.DataFrame()
    for factor in factors:
        fr = pd.concat([dfsp[factor], dfs[factor]], axis=1)
        fr.columns = ["percentile", "min"]
        fr = fr.groupby("percentile").min()
        fr = fr.assign(max=fr.shift(-1))
        fr.set_value(0, "min", -math.inf)
        fr.set_value(fr.index.values.max(), "max", math.inf)
        fr.index = pd.MultiIndex.from_product([[factor], fr.index.get_level_values(0)])
        factorRange = factorRange.append(fr)

    # PreProcess data (convert to z-scores)
    dfsp = dfsp.apply(axis=0, func=lambda x: stats.norm.ppf(0.005 + x))
    dfsp = dfsp.assign(y=dfs.y)
    factorRange = factorRange.assign(z=stats.norm.ppf(0.005 + factorRange.index.get_level_values(1)))

    # Create market data frame
    dfmp = dfsp.groupby(level=0).mean()[np.hstack([factors, "y"])]
    dfdp = dfsp.groupby(level=0).std()[np.hstack([factors, "y"])]
    dfdp.loc[:, "y"] = dfmp.y
    dfsp = dfsp.join(dfmp.loc[:, ["y"]], rsuffix="Market")

    # PreProcess data (impute)
    dfmp.fillna(0, inplace=True)
    dfsp.fillna(0, inplace=True)

    usedFactors = ["fundamental_8", "fundamental_15", "fundamental_18", "fundamental_43", "fundamental_45",
                   "fundamental_52", "fundamental_53", "fundamental_56", "fundamental_58", "fundamental_59",
                   "technical_13", "technical_20", "technical_22", "technical_30", "technical_34", "technical_40"]
    #marketFactors = ["fundamental_62", "fundamental_7", "technical_42", "technical_0", "technical_27", "technical_21"]
    #usedFactors = usedFactors + marketFactors

    # Create market model
    elnetModel_Market = createPredictionModel("ElasticNet - Market",
                                              lm.ElasticNetCV(l1_ratio=[0.01], alphas=[0.000504768968234]),
                                              dfmp, usedFactors, "y")

    # Create beta model
    calcBetas = pd.DataFrame(index=dfsp.index)
    calcBetas = calcBetas.assign(cummX=dfsp.groupby(level=1).yMarket.cumsum())
    calcBetas = calcBetas.assign(cummY=dfsp.groupby(level=1).y.cumsum())
    calcBetas = calcBetas.assign(XY=dfsp.yMarket * dfsp.y)
    calcBetas = calcBetas.assign(X2=dfsp.yMarket ** 2)
    calcBetas = calcBetas.assign(cummXY=calcBetas.groupby(level=1).XY.cumsum())
    calcBetas = calcBetas.assign(cummX2=calcBetas.groupby(level=1).X2.cumsum())
    CBG = calcBetas.groupby(level=1)
    calcBetas = calcBetas.assign(next20DZeroBeta=(CBG.cummXY.shift(-19) - CBG.cummXY.shift(1)) /
                                                 (CBG.cummX2.shift(-19) - CBG.cummX2.shift(1)))
    calcBetas = calcBetas.assign(next20DBeta=(20.0 * (CBG.cummXY.shift(-19) - CBG.cummXY.shift(1)) -
                                              (CBG.cummX.shift(-19) - CBG.cummX.shift(1)) * (
                                              CBG.cummY.shift(-19) - CBG.cummY.shift(1))) /
                                             (20.0 * (CBG.cummX2.shift(-19) - CBG.cummX2.shift(1)) -
                                              (CBG.cummX.shift(-19) - CBG.cummX.shift(1)) ** 2))
    dfsp = dfsp.assign(next20DZeroBeta=calcBetas.next20DZeroBeta, next20DBeta=calcBetas.next20DBeta)

    dfspA = dfsp[dfs.technical_22 < 0]
    dfspB = dfsp[dfs.technical_22 == 0]
    dfspC = dfsp[dfs.technical_22 > 0]
    elnetModel_BetaA = createPredictionModel("ElasticNet - Beta - A",
                                             lm.ElasticNetCV(l1_ratio=[0.99], alphas=[0.000189633948432]),
                                             dfspA, usedFactors, "next20DZeroBeta")
    elnetModel_BetaB = createPredictionModel("ElasticNet - Beta - B",
                                             lm.ElasticNetCV(l1_ratio=[0.99], alphas=[0.0001634045351]),
                                             dfspB, usedFactors, "next20DZeroBeta")
    elnetModel_BetaC = createPredictionModel("ElasticNet - Beta - C",
                                             lm.ElasticNetCV(l1_ratio=[0.99], alphas=[0.000348812551586]),
                                             dfspC, usedFactors, "next20DZeroBeta")
    dfsp = dfsp.assign(alpha=dfsp.y - dfsp.yMarket * dfsp.next20DZeroBeta)

    # Create dispersion model
    dfmp = dfmp.assign(dispersion=dfsp.assign(dev=abs(dfsp.alpha) * math.sqrt(math.pi / 2)).groupby(level=0).dev.mean())
    elnetModel_Dispersion = createPredictionModel("ElasticNet - Dispersion",
                                                  lm.ElasticNetCV(l1_ratio=[0.99], alphas=[1.65654786154e-06]),
                                                  dfmp, usedFactors, "dispersion")
    dfsp = dfsp.join(dfmp.loc[:, ["dispersion"]])

    # Create volatility model
    dfsp = dfsp.assign(relDev=abs(dfsp.alpha) * math.sqrt(math.pi / 2) / dfsp.dispersion)
    dfspA = dfsp[dfs.technical_22 < 0]
    dfspB = dfsp[dfs.technical_22 == 0]
    dfspC = dfsp[dfs.technical_22 > 0]
    elnetModel_VolatilityA = createPredictionModel("ElasticNet - Volatility - A",
                                                   lm.ElasticNetCV(
                                                       l1_ratio=[0.99], alphas=[4.57668989558e-05]),
                                                   dfspA, usedFactors, "relDev")
    elnetModel_VolatilityB = createPredictionModel("ElasticNet - Volatility - B",
                                                   lm.ElasticNetCV(
                                                       l1_ratio=[0.99], alphas=[9.21495319242e-05]),
                                                   dfspB, usedFactors, "relDev")
    elnetModel_VolatilityC = createPredictionModel("ElasticNet - Volatility - C",
                                                   lm.ElasticNetCV(
                                                       l1_ratio=[0.99], alphas=[0.00013627990326]),
                                                   dfspC, usedFactors, "relDev")
    dfsp.loc[dfs.technical_22 < 0, "relVolatility"] = elnetModel_VolatilityA.predict(
        dfsp.loc[dfs.technical_22 < 0, usedFactors])
    dfsp.loc[dfs.technical_22 == 0, "relVolatility"] = elnetModel_VolatilityB.predict(
        dfsp.loc[dfs.technical_22 == 0, usedFactors])
    dfsp.loc[dfs.technical_22 > 0, "relVolatility"] = elnetModel_VolatilityC.predict(
        dfsp.loc[dfs.technical_22 > 0, usedFactors])
    del dfspA
    del dfspB
    del dfspC
    gc.collect()

    # Create alpha model
    dfsp = dfsp.assign(relAlpha=dfsp.alpha / dfsp.relVolatility / dfsp.dispersion)
    elnetModel_Alpha = createPredictionModel("ElasticNet - Alpha",
                                             lm.ElasticNetCV(l1_ratio=[0.01], alphas=[0.01349897329]),
                                             dfsp, usedFactors, "relAlpha")
    dfsp = dfsp.assign(relAlpha=elnetModel_Alpha.predict(dfsp[usedFactors]))
    dfsp = dfsp.assign(yPrediction=dfsp.dispersion * dfsp.relVolatility * dfsp.alpha)

    print("Starting Test")

    rewards = []

    # Iterate through test set
    while True:
        target = observation.target
        timestamp = observation.features["timestamp"][0]
        # if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

        dfp = pd.DataFrame(index=observation.features.id)

        # Normalise the factors
        for factor in usedFactors:
            raw = np.array(observation.features[factor])
            new = np.array([float("NAN")] * raw.size)
            for i, fr in factorRange.loc[factor].iterrows():
                new[(raw >= fr["min"]) & (raw < fr["max"])] = fr.z
                dfp[factor] = new
        dfp.fillna(0, inplace=True)

        # Create Market
        dfm = pd.DataFrame(dfp.mean()).transpose()
        dfd = pd.DataFrame(dfp.std()).transpose()

        # Create y predictions
        market = elnetModel_Market.predict(dfd[usedFactors])
        beta = pd.Series([float("NAN")] * dfp.shape[0])
        beta[observation.features.technical_22 < 0] = elnetModel_BetaA.predict(
            dfp.loc[np.array(observation.features.technical_22 < 0), usedFactors])
        beta[observation.features.technical_22 == 0] = elnetModel_BetaB.predict(
            dfp.loc[np.array(observation.features.technical_22 == 0), usedFactors])
        beta[observation.features.technical_22 > 0] = elnetModel_BetaC.predict(
            dfp.loc[np.array(observation.features.technical_22 > 0), usedFactors])
        dispersion = elnetModel_Dispersion.predict(dfm[usedFactors])
        relVolatility = pd.Series([float("NAN")] * dfp.shape[0])
        relVolatility[observation.features.technical_22 < 0] = elnetModel_VolatilityA.predict(
            dfp.loc[np.array(observation.features.technical_22 < 0), usedFactors])
        relVolatility[observation.features.technical_22 == 0] = elnetModel_VolatilityB.predict(
            dfp.loc[np.array(observation.features.technical_22 == 0), usedFactors])
        relVolatility[observation.features.technical_22 > 0] = elnetModel_VolatilityC.predict(
            dfp.loc[np.array(observation.features.technical_22 > 0), usedFactors])
        relAlpha = elnetModel_Alpha.predict(dfp[usedFactors])
        target.loc[:, 'y'] = 0.0000 + dispersion * relVolatility * relAlpha

        # We perform a "step" by making our prediction and getting back an updated "observation":
        observation, reward, done, info = env.step(target)

        rewards.append(reward)
        print(reward)

        if done:
            print("Public score: {}".format(info["public_score"]))
            break