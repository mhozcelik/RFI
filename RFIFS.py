import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
    
def RFI(X_train,y_train,importance_type='gain'):
    # Relative Feature Importance
    RFI = pd.DataFrame(X_train.columns,columns=['feature'])
    
    selector = LGBMClassifier(random_state=42,verbosity=-1,importance_type=importance_type)
    selector.fit(X_train,y_train)
    RFI['feature_importance'] = selector.feature_importances_ / sum(selector.feature_importances_)

    my_corr = X_train.corr(method='pearson').abs()
    my_corr = my_corr * abs(np.eye(len(my_corr))-1)
    RFI['maximum_correlation'] = my_corr.max(axis=1).to_list()

    RFI['relative_feature_importance'] = RFI['feature_importance'].div(RFI['maximum_correlation'])
    RFI = RFI.set_index(['feature'])
    return RFI
