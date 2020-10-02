def read_data_():
    from pandas import read_csv

    train, test = (
        read_csv("/kaggle/input/titanic/train.csv"),
        read_csv("/kaggle/input/titanic/test.csv"),
    )

    _ = (
        train.set_index("PassengerId", inplace=True, verify_integrity=True),
        test.set_index("PassengerId", inplace=True, verify_integrity=True),
    )

    _ = train.sort_index(inplace=True), test.sort_index(inplace=True)
    return train, test


def get_title_feature_(train, test):
    train["Title"], test["Title"] = (
        train["Name"].apply(lambda x: x.split(", ")[1].split(".")[0]),
        test["Name"].apply(lambda x: x.split(", ")[1].split(".")[0]),
    )

    arg = {
        "Capt": "noble",
        "Col": "noble",
        "Don": "noble",
        "Dona": "noble",
        "Dr": "noble",
        "Jonkheer": "noble",
        "Lady": "noble",
        "Major": "noble",
        "Master": "master",
        "Miss": "miss",
        "Mlle": "miss",
        "Mme": "mrs",
        "Mr": "mr",
        "Mrs": "mrs",
        "Ms": "miss",
        "Rev": "noble",
        "Sir": "noble",
        "the Countess": "noble",
    }

    train["Title"], test["Title"] = (
        train["Title"].map(arg).astype("category"),
        test["Title"].map(arg).astype("category"),
    )
    return train, test


def get_surname_(train, test):
    train["Surname"], test["Surname"] = (
        train["Name"].str.split(", ", n=1, expand=True)[0],
        test["Name"].str.split(", ", n=1, expand=True)[0],
    )

    train["Surname"], test["Surname"] = (
        train["Surname"].str.replace("'", "").str.strip().str.lower(),
        test["Surname"].str.replace("'", "").str.strip().str.lower(),
    )
    return train, test


def get_first_names_(train, test):
    func = (
        lambda x: x.str.split(", ", n=1, expand=True)[1]
        .str.split(". ", n=1, expand=True)[1]
        .str.split('"', n=1, expand=True)[0]
        .str.split("(", n=1, expand=True)[0]
        .str.strip()
        .str.lower()
    )

    train["Name"], test["Name"] = func(train["Name"]), func(test["Name"])
    return train, test


def get_married_feature_(train, test):
    from pandas import concat, DataFrame

    df = (
        DataFrame(
            concat(
                [
                    train["Name"] + " " + train["Surname"],
                    test["Name"] + " " + test["Surname"],
                ]
            )
        )
        .groupby(0)[0]
        .transform("count")
        - 1
    )

    df[df > 1] = 1
    df[445] = 0  # "Dodge, Master. Washington"
    df[556] = 1  # "Duff Gordon, Lady."
    df[599] = 1  # "Duff Gordon, Sir. Cosmo Edmund"

    train["Married"], test["Married"] = (
        df.iloc[: train.shape[0]],
        df.iloc[train.shape[0] :],
    )
    return train, test


def get_family_counts_(train, test):
    from pandas import concat, DataFrame

    df = (
        DataFrame(concat([train["Surname"], test["Surname"]]))
        .groupby("Surname")["Surname"]
        .transform("count")
    )

    train["Family"], test["Family"] = (
        df.iloc[: train.shape[0]],
        df.iloc[train.shape[0] :],
    )
    return train, test


def extract_ticket_prefix_(train, test):
    from re import search

    func = lambda x: x.apply(
        lambda x: ""
        if search(r"(^.+(?= [0-9]+$))|(^[\./A-Za-z]+$)", str(x)) == None
        else search(r"(^.+(?= [0-9]+$))|(^[\./A-Za-z]+$)", str(x))
        .group(0)
        .replace("/", "")
        .replace(".", "")
        .replace(" ", "")
        .lower()
    )

    train["Prefix"], test["Prefix"] = (
        func(train["Ticket"]).astype("category"),
        func(test["Ticket"]).astype("category"),
    )
    return train, test


def extract_ticket_number_(train, test):
    from re import search

    func = lambda x: x.apply(
        lambda x: 0
        if search(r"(?![ ^])[0-9]+$", str(x)) == None
        else int(search(r"(?![ ^])[0-9]+$", str(x)).group(0).strip())
    )

    train["Number"], test["Number"] = func(train["Ticket"]), func(test["Ticket"])
    return train, test


def get_ticket_length_(train, test):
    func = lambda x: x.apply(lambda x: len(str(x)))

    train["Length"], test["Length"] = func(train["Number"]), func(test["Number"])

    train.loc[train["Ticket"] == "LINE", "Length"] = 0
    return train, test


def get_ticket_counts_(train, test):
    from pandas import concat, DataFrame

    df = (
        DataFrame(concat([train["Number"], test["Number"]]))
        .groupby("Number")["Number"]
        .transform("count")
    )

    train["Counts"], test["Counts"] = (
        df.iloc[: train.shape[0]],
        df.iloc[train.shape[0] :],
    )
    return train, test


def drop_unused_(train, test):
    _ = (
        train.drop(columns=["Name", "Ticket", "Surname", "Number"], inplace=True),
        test.drop(columns=["Name", "Ticket", "Surname", "Number"], inplace=True),
    )
    return train, test


def clean_data_(train, test):
    train["Embarked"].fillna("S", inplace=True)

    _ = (
        train["Cabin"].fillna("U", inplace=True),
        test["Cabin"].fillna("U", inplace=True),
    )

    train["Cabin"], test["Cabin"] = (
        train["Cabin"].apply(lambda x: x[0]).astype("category"),
        test["Cabin"].apply(lambda x: x[0]).astype("category"),
    )
    return train, test


def format_categories_(train, test):
    category = ["Embarked"]

    train[category], test[category] = (
        train[category].astype("category"),
        test[category].astype("category"),
    )

    _ = (
        train["Sex"].replace(["male", "female"], [0, 1], inplace=True),
        test["Sex"].replace(["male", "female"], [0, 1], inplace=True),
    )
    return train, test


def get_X_y_(train, test):
    X, X_submit = train.drop(columns="Survived"), test
    y = train["Survived"].to_numpy().ravel()
    return X, X_submit, y


def get_dummies_(X, X_submit):
    from pandas import concat, get_dummies
    from sklearn.preprocessing import MinMaxScaler

    df = concat([X, X_submit])

    for c in df.columns[(df.dtypes == int) | (df.dtypes == object)]:
        df[c] = df[c].astype("category")

    df = get_dummies(df)
    df.drop(columns="Cabin_T", inplace=True)
    X, X_submit = df.iloc[: X.shape[0]], df.iloc[X.shape[0] :]

    del df

    mm = MinMaxScaler().fit(concat([X, X_submit]))
    X.iloc[:, :], X_submit.iloc[:, :] = mm.transform(X), mm.transform(X_submit)
    return X, X_submit


def encode_categories_(X, X_submit, y):
    from pandas import concat, DataFrame
    from sklearn.model_selection import StratifiedKFold
    from category_encoders.cat_boost import CatBoostEncoder
    from sklearn.preprocessing import StandardScaler

    df = concat([X, X_submit])

    for c in df.columns[(df.dtypes == int) | (df.dtypes == object)]:
        df[c] = df[c].astype("category")

    X, X_submit = df.iloc[: X.shape[0]], df.iloc[X.shape[0] :]

    del df

    oof = DataFrame()

    for ix, ox in StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(
        X, y
    ):
        oof = oof.append(
            CatBoostEncoder(handle_missing="return_nan", random_state=0)
            .fit(X.iloc[ix, :], y[ix])
            .transform(X.iloc[ox, :]),
            ignore_index=False,
        )

    X_submit = (
        CatBoostEncoder(handle_missing="return_nan", random_state=0)
        .fit(X, y)
        .transform(X_submit)
    )

    X = oof.sort_index()

    sc = StandardScaler().fit(concat([X, X_submit]))
    X.iloc[:, :], X_submit.iloc[:, :] = sc.transform(X), sc.transform(X_submit)
    return X, X_submit


def impute_missing_(X, X_submit):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from pandas import concat

    ii = IterativeImputer(
        initial_strategy="median", imputation_order="random", random_state=0
    ).fit(concat([X, X_submit]))
    X.iloc[:, :], X_submit.iloc[:, :] = ii.transform(X), ii.transform(X_submit)
    return X, X_submit


def get_decomposition_features_(X, X_submit):
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP
    from pandas import concat
    from sklearn.decomposition import KernelPCA

    u = StandardScaler().fit_transform(
        UMAP(n_components=1, random_state=0).fit_transform(concat([X, X_submit]))
    )

    k = StandardScaler().fit_transform(
        KernelPCA(
            n_components=1, kernel="rbf", gamma=0.0496, random_state=0, n_jobs=-1
        ).fit_transform(concat([X, X_submit]))
    )

    X["UMAP"], X_submit["UMAP"] = u[: X.shape[0]], u[X.shape[0] :]
    X["KernelPCA"], X_submit["KernelPCA"] = k[: X.shape[0]], k[X.shape[0] :]
    return X, X_submit
