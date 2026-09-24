import pandas as pd

def preprocess_input(data_dict):
    """Preprocessing untuk 1 pelanggan baru dari input Streamlit."""
    df_new = pd.DataFrame([data_dict])
    
    # Feature engineering
    df_new['Age'] = 2024 - df_new['Year_Birth']
    spending_cols = ['MntWines','MntFruits','MntMeatProducts',
                     'MntFishProducts','MntSweetProducts','MntGoldProds']
    df_new['TotalSpending'] = df_new[spending_cols].sum(axis=1)
    purchase_cols = ['NumWebPurchases','NumCatalogPurchases',
                     'NumStorePurchases','NumDealsPurchases']
    df_new['TotalPurchases'] = df_new[purchase_cols].sum(axis=1)
    df_new['Dependents'] = df_new['Kidhome'] + df_new['Teenhome']
    cmp_cols = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3',
                'AcceptedCmp4','AcceptedCmp5']
    df_new['TotalCampaignsAccepted'] = df_new[cmp_cols].sum(axis=1)
    df_new['SpendingRatio'] = df_new['TotalSpending'] / (df_new['Income'] + 1)
    df_new['DaysSinceEnrollment'] = (pd.Timestamp.now() - 
        pd.to_datetime(df_new['Dt_Customer'], dayfirst=True)).dt.days
    
    features = ['Income','Recency','TotalSpending','TotalPurchases','Age',
                'Dependents','TotalCampaignsAccepted','NumWebVisitsMonth',
                'SpendingRatio','DaysSinceEnrollment']
    return df_new[features]