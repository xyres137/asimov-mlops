def feature_engineering(X):
    X = X.copy()
    X["COT_CER_Ratio"] = X["Value COT"] / (X["Value CER"] + 1e-6)  #
    X["CWP_Squared"] = X["Value CWP"] ** 2
    return X


def get_feature_engineering_transformer():
    from sklearn.preprocessing import FunctionTransformer
    return FunctionTransformer(feature_engineering)


def classifier(value_counts):
    def classify_bin(value):
        frequency = value_counts.get(value, 0)
        if 2 <= frequency <= 20:
            return "few"
        elif 20 < frequency <= 100:
            return "medium"
        elif frequency > 100:
            return "many"
        else:
            return "other"

    return classify_bin
