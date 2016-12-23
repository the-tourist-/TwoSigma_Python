# ToDo: Try out grouping on both technical_34 and technical_22
# ToDo: Investigate heirachial model with combination of 9 class group, both 3 class groups and 1 class group
# ToDo: On dfsp rescale each timestamp using dfmp and dfdp
# ToDo: Do better imputation, possibly by adding previous timestamp data for factor (maybe use fancyimpute)
# ToDo: Add flight/fight indicator to prediction - Weight stocks -1 to 1 by predicted volatility to create new factors (and y), and then predict the weighted y
# ToDo: Look into using moving averages of the factors
# ToDo: Use full dataset
# ToDo: Use other models in an ensemble approach
# ToDo: Reduce the number of factors used
# ToDo: Add polynomials

if __name__ == '__main__':

    testing = True
    full = True

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
            if (testing):
                rScore = kagglegym.r_score(filteredData[outcome], predictions)
                print("Training R Score: %s" % "{0:.3%}".format(rScore))
        else:
            print("Model Rejected - No Coefficients")

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
            rScore = kagglegym.r_score(filteredData[outcome], cvPredictions)
            print("C-V R Score: %s" % "{0:.3%}".format(rScore))
            model.cvPredictions = cvPredictions

        model.coefs = coefs

        return model


    # Asset Class, used for grouping by technical_22 (and possibly technical_34)
    class assetClass:
        'Groups the assets into separate classes with their own unique characteristics'

        def __init__(self):
            None

    assetClasses = [assetClass() for i in range(3)]

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

    # The "environment" is our interface for code competitions
    env = kagglegym.make()

    # We get our initial observation by calling "reset"
    observation = env.reset()
    print("Train has {} rows".format(len(observation.train)))
    print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

    # Load data frame
    df = observation.train
    df.set_index(["timestamp", "id"], inplace = True)
    factors = [factor for factor in df.columns.values if not factor in ["id", "timestamp", "y"]]

    class22Values = np.array(range(3)) / 2 - 0.5

    for class22 in class22Values:
        dfs = df[df.technical_22 == class22]

        # Create small data frame
        random.seed()
        if ~full & testing:
            smallSize = 75000
            timestamps = list(dfs.index.get_level_values(0).unique().values)
            timestamps = [ timestamps[i] for i in sorted(random.sample(range(len(timestamps)),
                                                                       round(math.sqrt(smallSize / dfs.shape[0]) * len(timestamps)))) ]
            dfs = dfs.loc[timestamps]
            ids = list(dfs.index.get_level_values(1).unique().values)
            ids = [ ids[i] for i in sorted(random.sample(range(len(ids)), round(smallSize / dfs.shape[0] * len(ids)))) ]
            dfs = dfs.loc[dfs.index.get_level_values(1).isin(ids)]

        print("\nTECHNICAL 22 = ", class22, dfs.shape[0], "\n")

        # PreProcess data (normalise to percentile)
        dfsp = 99 * dfs[factors].rank(method = "average", na_option = "keep", pct = True) // 1 / 100.0

        # create factor range z-score lookup
        factorRange = pd.DataFrame()
        for factor in factors:
            print(factor)
            fr = pd.concat([dfsp[factor], dfs[factor]], axis = 1)
            fr.columns = ["percentile", "min"]
            fr = fr.groupby("percentile").min()
            fr = fr.assign(max = fr.shift(-1))
            fr.set_value(0, "min", -math.inf)
            fr.set_value(fr.index.values.max(), "max", math.inf)
            fr.index = pd.MultiIndex.from_product([[factor], fr.index.get_level_values(0)])
            factorRange = factorRange.append(fr)

        # PreProcess data (convert to z-scores)
        dfsp = dfsp.apply(axis = 0, func = lambda x: stats.norm.ppf(0.005 + x))
        dfsp = dfsp.assign(y = dfs.y)
        factorRange = factorRange.assign(z = stats.norm.ppf(0.005 + factorRange.index.get_level_values(1)))

        # Create market data frame
        dfmp = dfsp.groupby(level = 0).mean()[np.hstack([factors, "y"])]
        dfdp = dfsp.groupby(level = 0).std()[np.hstack([factors, "y"])]

        # PreProcess data (impute)
        dfmp.fillna(0, inplace = True)
        dfsp.fillna(0, inplace = True)

        usedFactors = ["fundamental_0", "fundamental_6", "fundamental_8", "fundamental_23", "fundamental_53", "fundamental_55", "fundamental_60",
                      "technical_7", "technical_13", "technical_20", "technical_22", "technical_29", "technical_30", "technical_34", "technical_35", "technical_40"]
        usedFactors = [factor for factor in usedFactors if factor != "technical_22"]

        # Create simple model
        # elnetModel_Y = createPredictionModel("ElasticNet - Y",
        #                                      lm.ElasticNetCV(l1_ratio = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
        #                                      dfsp, factors, "y")

        # Create market model
        elnetModel_Market = createPredictionModel("ElasticNet - Market",
                                                  lm.ElasticNetCV(l1_ratio = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                                  dfmp, usedFactors, "y")
        dfsp = dfsp.join(dfmp.loc[:, ["y"]], rsuffix = "Market")

        # Create beta model
        calcBetas = pd.DataFrame(index = dfsp.index)
        calcBetas = calcBetas.assign(cummX = dfsp.groupby(level = 1).yMarket.cumsum())
        calcBetas = calcBetas.assign(cummY = dfsp.groupby(level = 1).y.cumsum())
        calcBetas = calcBetas.assign(XY = dfsp.yMarket * dfsp.y)
        calcBetas = calcBetas.assign(X2 = dfsp.yMarket ** 2)
        calcBetas = calcBetas.assign(cummXY = calcBetas.groupby(level = 1).XY.cumsum())
        calcBetas = calcBetas.assign(cummX2 = calcBetas.groupby(level = 1).X2.cumsum())
        CBG = calcBetas.groupby(level = 1)
        calcBetas = calcBetas.assign(next20DZeroBeta = (CBG.cummXY.shift(-19) - CBG.cummXY.shift(1)) /
                                                       (CBG.cummX2.shift(-19) - CBG.cummX2.shift(1)))
        calcBetas = calcBetas.assign(next20DBeta = (20.0 * (CBG.cummXY.shift(-19) - CBG.cummXY.shift(1)) -
                                                    (CBG.cummX.shift(-19) - CBG.cummX.shift(1)) * (CBG.cummY.shift(-19) - CBG.cummY.shift(1))) /
                                                   (20.0 * (CBG.cummX2.shift(-19) - CBG.cummX2.shift(1)) -
                                                    (CBG.cummX.shift(-19) - CBG.cummX.shift(1)) ** 2))
        dfsp = dfsp.assign(next20DZeroBeta = calcBetas.next20DZeroBeta, next20DBeta = calcBetas.next20DBeta)

        elnetModel_Beta = createPredictionModel("ElasticNet - Beta",
                                                lm.ElasticNetCV(l1_ratio = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                                dfsp, usedFactors, "next20DZeroBeta")
        dfsp = dfsp.assign(alpha = dfsp.y - dfsp.yMarket * dfsp.next20DZeroBeta)

        # Create dispersion model
        dfmp = dfmp.assign(dispersion = dfsp.assign(dev = abs(dfsp.alpha) * math.sqrt(math.pi / 2)).groupby(level=0).dev.mean())
        elnetModel_Dispersion = createPredictionModel("ElasticNet - Dispersion",
                                                      lm.ElasticNetCV(l1_ratio = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                                      dfmp, usedFactors, "dispersion")
        dfsp = dfsp.join(dfmp.loc[:, ["dispersion"]])

        # Create volatility model
        elnetModel_Volatility = createPredictionModel("ElasticNet - Volatility",
                                                      lm.ElasticNetCV(l1_ratio=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                                      dfsp.assign(relDev = abs(dfsp.alpha) * math.sqrt(math.pi / 2) / dfsp.dispersion),
                                                      usedFactors, "relDev")
        dfsp = dfsp.assign(relVolatility = elnetModel_Volatility.predict(dfsp[usedFactors]))

        # Create alpha model
        elnetModel_Alpha = createPredictionModel("ElasticNet - Alpha",
                                                 lm.ElasticNetCV(l1_ratio=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
                                                 dfsp.assign(relAlpha = dfsp.alpha / dfsp.relVolatility / dfsp.dispersion),
                                                 usedFactors, "relAlpha")
        dfsp = dfsp.assign(relAlpha = elnetModel_Alpha.predict(dfsp[usedFactors]))

        # Save everything to assetClass
        assetClasses[int(class22 * 2 + 1)].dfs = dfs
        assetClasses[int(class22 * 2 + 1)].dfsp = dfsp
        assetClasses[int(class22 * 2 + 1)].dfmp = dfmp
        assetClasses[int(class22 * 2 + 1)].dfdp = dfdp
        assetClasses[int(class22 * 2 + 1)].factorRange = factorRange
        assetClasses[int(class22 * 2 + 1)].elnetModel_Market = elnetModel_Market
        assetClasses[int(class22 * 2 + 1)].elnetModel_Beta = elnetModel_Beta
        assetClasses[int(class22 * 2 + 1)].elnetModel_Dispersion = elnetModel_Dispersion
        assetClasses[int(class22 * 2 + 1)].elnetModel_Volatility = elnetModel_Volatility
        assetClasses[int(class22 * 2 + 1)].elnetModel_Alpha = elnetModel_Alpha

    print("Starting Test")

    rewards = list()

    # Iterate through test set
    while True:
        target = observation.target
        timestamp = observation.features["timestamp"][0]

        for class22 in class22Values:
            dfc = observation.features.loc[observation.features.technical_22 == class22]
            dfp = pd.DataFrame(index=dfc.id)

            # Normalise the factors
            for factor in usedFactors:
                raw = np.array(dfc[factor])
                new = np.array([float("NAN")] * raw.size)
                for i, fr in assetClasses[int(class22 * 2 + 1)].factorRange.loc[factor].iterrows():
                    new[(raw >= fr["min"]) & (raw < fr["max"])] = fr.z
                dfp[factor] = new
            dfp.fillna(0, inplace=True)

            # Create Market
            dfm = pd.DataFrame(dfp.mean()).transpose()

            ac = assetClasses[int(class22 * 2 + 1)]

            # Create y predictions
            market = ac.elnetModel_Market.predict(dfm[usedFactors])
            beta = ac.elnetModel_Beta.predict(dfp[usedFactors])
            dispersion = ac.elnetModel_Dispersion.predict(dfm[usedFactors])
            relVolatility = ac.elnetModel_Volatility.predict(dfp[usedFactors])
            relAlpha = ac.elnetModel_Alpha.predict(dfp[usedFactors])
            target.loc[observation.features.technical_22 == class22, 'y'] = dispersion * relVolatility * relAlpha

        # We perform a "step" by making our prediction and getting back an updated "observation":
        observation, reward, done, info = env.step(target)

        rewards.append(reward)

        # if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp), reward)

        if done:
            print("Public score: {}".format(info["public_score"]))
            if testing:
                pd.DataFrame(rewards).to_clipboard()
            break