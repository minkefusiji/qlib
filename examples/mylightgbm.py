import lightgbm as lgb

def get_dataset(x_train, y_train, x_test, y_test, x_valid, y_valid):
    lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(x_test, y_test, free_raw_data=False)
    lgb_valid = lgb.Dataset(x_valid, y_valid, reference=lgb_train, free_raw_data=False)

    return lgb_train, lgb_eval, lgb_valid

