# ToDo: Impute using technical_22 and technical_34
# ToDo: Add polynomials for Volatility and  Beta
# ToDo: Make CV based on regime changes
# ToDo: Used ranking not scaling for timestamp normalisation
# TODo: Make relAlpha a mixed model effectively by adding 3 more sets of factors which only have values for the 3 technical_22 values
# ToDo: Use both timestamp and global values in models
# ToDo: Go to having both time stamped and non-timestamped scaling
# ToDo: Add flight/fight indicator to prediction - Weight stocks -1 to 1 by predicted volatility to create new factors (and y), and then predict the weighted y
# ToDo: Look into using moving averages of the factors
# ToDo: Use full dataset
# ToDo: Use other models in an ensemble approach
# ToDo: Reduce the number of factors used
# ToDo: In final submission, submit two models, one with training mean market return, and one 0.00008 higher.

def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight,
                  multioutput=multioutput)
    r = (np.sign(r2) * np.sqrt(np.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r


def createPredictionModel(name, model, data, factors, outcome, includePredictions):
    print("\n" + name)

    cvFolds = GroupKFold(n_splits=5)

    filteredData = data[~data[outcome].isnull()]
    groupDivisor = math.ceil((data.index.get_level_values(0).max() - data.index.get_level_values(0).min() + 1) / 5)
    cvGroups = filteredData.index.get_level_values(0) // groupDivisor

    params = model.get_params()
    params["n_jobs"] = (1, -1)[testing]
    params["cv"] = cvFolds.split(filteredData, groups=cvGroups)
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
        print("\nIntercept: {0:.6f}\n".format(model.intercept_))
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

    if includePredictions:
        # Perform CV
        cvmodel = lm.ElasticNetCV(l1_ratio=model.l1_ratio_, alphas=[model.alpha_], n_jobs=1)  # (1, -1)[testing])
        cvFolds = GroupKFold(n_splits=5)
        cvPredictions = ms.cross_val_predict(cvmodel, filteredData[factors], filteredData[outcome],
                                             cv=cvFolds.split(filteredData, groups=cvGroups),
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

    usedFactors = ["derived_1", "fundamental_0", "fundamental_8", "fundamental_18", "fundamental_45", "fundamental_50",
                   "fundamental_53", "fundamental_56", "fundamental_58",
                   "technical_7", "technical_20", "technical_22", "technical_30",
                   "technical_34", "technical_35", "technical_36", "technical_40"]
    marketFactors = ["fundamental_62", "fundamental_7", "technical_42", "technical_0", "technical_27", "technical_21"]
    #usedFactors = usedFactors + marketFactors

    # FIX
    #usedFactors = factors

    usedFactorsSqr = ["{}_Sqr".format(f) for f in usedFactors]

    dfs = df[usedFactors + ["y"]]

    del df
    gc.collect()

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

    # Impute missing values in dfs
    imputeLookup = dfs[usedFactors].groupby(by=["technical_22", "technical_34"]).median()
    dfi = dfs.reset_index().set_index(keys=["technical_22", "technical_34"], drop=False).join(imputeLookup, lsuffix="_orig")
    dfi.set_index(keys=["timestamp", "id"], inplace=True)
    dfs = dfs[usedFactors].fillna(dfi[usedFactors]).join(dfs.y)

    del dfi
    gc.collect()

    # PreProcess data (normalise to percentile)
    dfsp = 99 * dfs[usedFactors].rank(method="average", na_option="keep", pct=True) // 1 / 100.0

    # create factor range z-score lookup
    factorRange = pd.DataFrame()
    for factor in usedFactors:
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
    factorRange = factorRange.assign(z=stats.norm.ppf(0.005 + factorRange.index.get_level_values(1)))

    # Create market data frame
    dfsp = dfsp.assign(y=dfs.y)
    dfmp = dfsp.groupby(level=0).mean()[usedFactors + ["y"]]
    dfdp = dfsp.groupby(level=0).std()[usedFactors + ["y"]]
    dfmp.fillna(0.0, inplace=True)
    dfdp.fillna(1.0, inplace=True)
    for factor in dfdp.columns[dfdp[usedFactors].min() == 0]:
        dfdp.loc[dfdp[factor] == 0, factor] = 1.0

    # PreProcess data - center, scale dfsp by timestamp, impute
    dfsp = (dfsp[usedFactors] - dfmp[usedFactors]) / dfdp[usedFactors]
    dfsp = dfsp.assign(y=dfs.y)
    dfsp = dfsp.join(dfmp.loc[:, ["y"]], rsuffix="Market")

    # Create Sqr Version of dfsp
    dfsp2 = pd.DataFrame((dfsp[usedFactors] ** 2 - 1.0) / math.sqrt(2.0))
    dfsp2.columns = usedFactorsSqr

    if testing:
        # Create market model
        elnetModel_Market = createPredictionModel("ElasticNet - Market",
                                                  lm.ElasticNetCV(l1_ratio=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                                  dfmp, usedFactors, "y", True)
        dfmp.loc[:, "yMarketPrediction"] = elnetModel_Market.cvPredictions
        dfsp = dfsp.join(dfmp.loc[:, ["yMarketPrediction"]])

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
    dfspi = pd.concat([dfsp, dfsp2], axis=1, ignore_index=True)
    dfspi.columns = list(dfsp.columns) + list(dfsp2.columns)

    dfspA = dfspi[dfs.technical_22 < 0]
    dfspB = dfspi[dfs.technical_22 == 0]
    dfspC = dfspi[dfs.technical_22 > 0]

    elnetModel_BetaA = createPredictionModel("ElasticNet - Beta - A",
                                             lm.ElasticNetCV(l1_ratio=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                             dfspA, usedFactors + usedFactorsSqr, "next20DZeroBeta", includePredictions=testing)
    elnetModel_BetaB = createPredictionModel("ElasticNet - Beta - B",
                                             lm.ElasticNetCV(l1_ratio=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                             dfspB, usedFactors + usedFactorsSqr, "next20DZeroBeta", includePredictions=testing)
    elnetModel_BetaC = createPredictionModel("ElasticNet - Beta - C",
                                             lm.ElasticNetCV(l1_ratio=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                             dfspC, usedFactors + usedFactorsSqr, "next20DZeroBeta", includePredictions=testing)
    dfsp = dfsp.assign(alpha=dfsp.y - dfsp.yMarket * dfsp.next20DZeroBeta)
    if testing:
        dfsp.loc[(dfs.technical_22 < 0) & (~dfsp.next20DZeroBeta.isnull()), "betaPrediction"] = \
            elnetModel_BetaA.cvPredictions
        dfsp.loc[(dfs.technical_22 < 0) & (dfsp.next20DZeroBeta.isnull()), "betaPrediction"] = \
            elnetModel_BetaA.predict(dfspA.loc[dfspA.next20DZeroBeta.isnull(), usedFactors + usedFactorsSqr])
        dfsp.loc[(dfs.technical_22 == 0) & (~dfsp.next20DZeroBeta.isnull()), "betaPrediction"] = \
            elnetModel_BetaB.cvPredictions
        dfsp.loc[(dfs.technical_22 == 0) & (dfsp.next20DZeroBeta.isnull()), "betaPrediction"] = \
            elnetModel_BetaB.predict(dfspB.loc[dfspB.next20DZeroBeta.isnull(), usedFactors + usedFactorsSqr])
        dfsp.loc[(dfs.technical_22 > 0) & (~dfsp.next20DZeroBeta.isnull()), "betaPrediction"] = \
            elnetModel_BetaC.cvPredictions
        dfsp.loc[(dfs.technical_22 > 0) & (dfsp.next20DZeroBeta.isnull()), "betaPrediction"] = \
            elnetModel_BetaC.predict(dfspC.loc[dfspC.next20DZeroBeta.isnull(), usedFactors + usedFactorsSqr])

    # Create dispersion model
    dfmp = dfmp.assign(dispersion=dfsp.assign(dev=abs(dfsp.alpha) * math.sqrt(math.pi / 2)).groupby(level=0).dev.mean())
    elnetModel_Dispersion = createPredictionModel("ElasticNet - Dispersion",
                                                  lm.ElasticNetCV(l1_ratio=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                                  dfmp, usedFactors, "dispersion", includePredictions=testing)
    dfsp = dfsp.join(dfmp.loc[:, ["dispersion"]])
    if testing:
        dfmp.loc[~dfmp.dispersion.isnull(), "dispersionPrediction"] = elnetModel_Dispersion.cvPredictions
        dfmp.loc[dfmp.dispersion.isnull(), "dispersionPrediction"] = elnetModel_Dispersion.predict(dfmp.loc[dfmp.dispersion.isnull(), usedFactors])
        dfsp = dfsp.join(dfmp.loc[:, ["dispersionPrediction"]])

    # Create volatility model
    dfsp = dfsp.assign(relDev=abs(dfsp.alpha) * math.sqrt(math.pi / 2) / dfsp.dispersion)
    dfspi = pd.concat([dfsp, dfsp2], axis=1, ignore_index=True)
    dfspi.columns = list(dfsp.columns) + list(dfsp2.columns)
    dfspA = dfspi[dfs.technical_22 < 0]
    dfspB = dfspi[dfs.technical_22 == 0]
    dfspC = dfspi[dfs.technical_22 > 0]
    elnetModel_VolatilityA = createPredictionModel("ElasticNet - Volatility - A",
                                                   lm.ElasticNetCV(l1_ratio=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                                   dfspA, usedFactors + usedFactorsSqr, "relDev", includePredictions=True)
    elnetModel_VolatilityB = createPredictionModel("ElasticNet - Volatility - B",
                                                   lm.ElasticNetCV(l1_ratio=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                                   dfspB, usedFactors + usedFactorsSqr, "relDev", includePredictions=True)
    elnetModel_VolatilityC = createPredictionModel("ElasticNet - Volatility - C",
                                                   lm.ElasticNetCV(l1_ratio=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                                   dfspC, usedFactors + usedFactorsSqr, "relDev", includePredictions=True)
    dfsp.loc[(dfs.technical_22 < 0) & (~dfsp.relDev.isnull()), "relVolatilityPrediction"] = \
        elnetModel_VolatilityA.cvPredictions
    dfsp.loc[(dfs.technical_22 < 0) & (dfsp.relDev.isnull()), "relVolatilityPrediction"] = \
        elnetModel_VolatilityA.predict(dfspA.loc[dfspA.relDev.isnull(), usedFactors + usedFactorsSqr])
    dfsp.loc[(dfs.technical_22 == 0) & (~dfsp.relDev.isnull()), "relVolatilityPrediction"] = \
        elnetModel_VolatilityB.cvPredictions
    dfsp.loc[(dfs.technical_22 == 0) & (dfsp.relDev.isnull()), "relVolatilityPrediction"] = \
        elnetModel_VolatilityB.predict(dfspB.loc[dfspB.relDev.isnull(), usedFactors + usedFactorsSqr])
    dfsp.loc[(dfs.technical_22 > 0) & (~dfsp.relDev.isnull()), "relVolatilityPrediction"] = \
        elnetModel_VolatilityC.cvPredictions
    dfsp.loc[(dfs.technical_22 > 0) & (dfsp.relDev.isnull()), "relVolatilityPrediction"] = \
        elnetModel_VolatilityC.predict(dfspC.loc[dfspC.relDev.isnull(), usedFactors + usedFactorsSqr])

    del dfspA
    del dfspB
    del dfspC
    #del dfspi
    gc.collect()

    # Create alpha model
    dfsp = dfsp.assign(relAlpha=dfsp.alpha / dfsp.relVolatilityPrediction / dfsp.dispersion)
    elnetModel_Alpha = createPredictionModel("ElasticNet - Alpha",
                                             lm.ElasticNetCV(l1_ratio=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                             dfsp, usedFactors, "relAlpha", includePredictions=testing)

    if testing:
        dfsp.loc[~dfsp.relAlpha.isnull(), "relAlphaPrediction"] = elnetModel_Alpha.cvPredictions
        dfsp.loc[dfsp.relAlpha.isnull(), "relAlphaPrediction"] = \
            elnetModel_Alpha.predict(dfsp.loc[dfsp.relAlpha.isnull(), usedFactors])

        # Create y predictions
        dfsp = dfsp.assign(yPredAlphaComponent = dfsp.dispersionPrediction * dfsp.relVolatilityPrediction * dfsp.relAlphaPrediction)
        dfsp = dfsp.assign(yPredMarketComponent = dfsp.yMarketPrediction * dfsp.betaPrediction)
        dfsp = dfsp.assign(yPredBetaComponent = elnetModel_Market.intercept_ * dfsp.betaPrediction)
        dfsp = dfsp.assign(yPrediction1 = dfsp.yPredAlphaComponent + dfsp.yPredMarketComponent)
        dfsp = dfsp.assign(yPrediction2 = dfsp.yPredAlphaComponent + dfsp.yPredBetaComponent)
        dfsp = dfsp.assign(yPrediction3 = dfsp.yPredAlphaComponent + elnetModel_Market.intercept_)

        print("\n\nAlpha R-Score: {0:0.2f}%".format(100 * r_score(dfsp.y, dfsp.yPredAlphaComponent)))
        print("Beta*0.02% R-Score: {0:0.2f}%".format(100 * r_score(dfsp.y, dfsp.yPredBetaComponent)))
        print("Market R-Score: {0:0.2f}%".format(100 * r_score(dfsp.y, dfsp.yPredMarketComponent)))
        print("Alpha+Constant*0.02% R-Score: {0:0.2f}%".format(100 * r_score(dfsp.y, dfsp.yPrediction3)))
        print("Alpha+Beta*0.02% R-Score: {0:0.2f}%".format(100 * r_score(dfsp.y, dfsp.yPrediction2)))
        print("Alpha+Market R-Score: {0:0.2f}%\n\n".format(100 * r_score(dfsp.y, dfsp.yPrediction1)))

    print("Starting Test\n")

    rewards = []

    if testing:
        env = kagglegym.make()
        observation = env.reset()

    # Iterate through test set
    while True:
        target = observation.target
        timestamp = observation.features["timestamp"][0]
        # if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

        dft = observation.features[["id"] + usedFactors]
        dfi = dft.set_index(keys=["technical_22", "technical_34"], drop=False).join(imputeLookup, lsuffix="_orig")
        dfi.set_index(keys=["id"], inplace=True)
        dft.set_index(keys=["id"], inplace=True)
        dft = dft[usedFactors].fillna(dfi[usedFactors])

        # Normalise the factors
        dfp = pd.DataFrame(index=dft.index)
        for factor in usedFactors:
            raw = np.array(dft[factor])
            new = np.array([float("NAN")] * raw.size)
            for i, fr in factorRange.loc[factor].iterrows():
                new[(raw >= fr["min"]) & (raw < fr["max"])] = fr.z
            dfp[factor] = new

        # Create Market
        dfm = pd.DataFrame(dfp.mean()).transpose()
        dfm.fillna(0, inplace=True)

        # Finish preprocessing
        sd = np.nan_to_num(dfp.std())
        sd = np.transpose([1 if x < 1e-12 else x for x in sd])
        mean = np.nan_to_num(dfp.mean())
        dfp = (dfp - mean) / sd

        # Create Squared Factors
        dfp2 = pd.DataFrame((dfp[usedFactors] ** 2 - 1.0) / math.sqrt(2.0))
        dfp2.columns = usedFactorsSqr
        dfpi = pd.concat([dfp, dfp2], axis=1, ignore_index=True)
        dfpi.columns = list(dfp.columns) + list(dfp2.columns)

        gc.collect()

        # Create y predictions
        if testing:
            market = elnetModel_Market.predict(dfm[usedFactors])
        beta = pd.Series([float("NAN")] * dfp.shape[0])
        beta[observation.features.technical_22 < 0] = elnetModel_BetaA.predict(
            dfpi.loc[np.array(observation.features.technical_22 < 0), usedFactors + usedFactorsSqr])
        beta[observation.features.technical_22 == 0] = elnetModel_BetaB.predict(
            dfpi.loc[np.array(observation.features.technical_22 == 0), usedFactors + usedFactorsSqr])
        beta[observation.features.technical_22 > 0] = elnetModel_BetaC.predict(
            dfpi.loc[np.array(observation.features.technical_22 > 0), usedFactors + usedFactorsSqr])
        dispersion = elnetModel_Dispersion.predict(dfm[usedFactors])
        relVolatility = pd.Series([float("NAN")] * dfp.shape[0])
        relVolatility[observation.features.technical_22 < 0] = elnetModel_VolatilityA.predict(
            dfpi.loc[np.array(observation.features.technical_22 < 0), usedFactors + usedFactorsSqr])
        relVolatility[observation.features.technical_22 == 0] = elnetModel_VolatilityB.predict(
            dfpi.loc[np.array(observation.features.technical_22 == 0), usedFactors + usedFactorsSqr])
        relVolatility[observation.features.technical_22 > 0] = elnetModel_VolatilityC.predict(
            dfpi.loc[np.array(observation.features.technical_22 > 0), usedFactors + usedFactorsSqr])
        relAlpha = elnetModel_Alpha.predict(dfp[usedFactors])
        target.loc[:, 'y'] = dispersion * relVolatility * relAlpha

        # Remove outliers and center
        target.loc[target.y > target.y.median() + target.y.mad() * 3 * math.sqrt(math.pi / 2), "y"] = target.y.median() + target.y.mad() * 3 * math.sqrt(math.pi / 2)
        target.loc[target.y < target.y.median() - target.y.mad() * 3 * math.sqrt(math.pi / 2), "y"] = target.y.median() - target.y.mad() * 3 * math.sqrt(math.pi / 2)
        target.loc[:, "y"] = target.loc[:, "y"] - target.y.mean()

        # Add constant market return
        target.loc[:, "y"] = target.loc[:, "y"] + 0

        # Perform a "step" by making our prediction and getting back an updated "observation":
        observation, reward, done, info = env.step(target)

        rewards.append(reward)
        print(reward)

        if done:
            print("Public score: {}".format(info["public_score"]))
            if testing:
                print("Correlation: {}".format(info["correlation"]))
            break