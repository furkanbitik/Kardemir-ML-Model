import pandas as pd
import numpy as np
import joblib
import warnings

# --- 0. Gerekli Kütüphaneleri Yükleme ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor

# Ayarlar ve Görüntüleme
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)
warnings.filterwarnings('ignore')


# --- Merkezi Ön İşleme Fonksiyonu ---
def preprocess_data(df, base_columns_info, medians):
    processed_df = df.copy()
    for col in processed_df.columns:
        if col != 'dokum_no':
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    missing_cols = set(base_columns_info) - set(processed_df.columns)
    for col in missing_cols:
        processed_df[col] = medians[col]
    processed_df.fillna(medians, inplace=True)
    return processed_df


# --- 1. Veri Yükleme ve Ön İşleme ---
print("--- 1. Veri Yükleme ve Ön İşleme ---")
try:
    df_raw = pd.read_csv('final_birlesik_veri2.csv', sep=';', decimal=',')
except FileNotFoundError:
    print("Hata: 'final_birlesik_veri2.csv' bulunamadı.")
    exit()

df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
df_clean = df_raw.drop(columns=['dokum_no', 'numune_no'], errors='ignore')
numeric_df = df_clean.apply(pd.to_numeric, errors='coerce')
training_medians = numeric_df.median()
joblib.dump(training_medians, 'training_medians.joblib')
print("Ön işleme tamamlandı.")

# --- 2. Veri Hazırlama ---
print("\n--- 2. Veri Hazırlama ---")
target_columns = ['cap', 'ovalite', 'elastikiyet', 'rel_alt_akma_dayanimi', 'reh_ust_akma_dayanimi', 'tufal_orani']
feature_columns = [col for col in df_clean.columns if col not in target_columns]
X = df_clean[feature_columns]
y = df_clean[target_columns]
joblib.dump(feature_columns, 'feature_columns.joblib')

y_transformed = np.log1p(y)
X_train, X_test, y_train_transformed, y_test_transformed = train_test_split(X, y_transformed, test_size=0.2,
                                                                            random_state=42)
y_test_original = y.loc[y_test_transformed.index]

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'minmax_scaler.joblib')
print("Veri ayırma, hedef dönüşümü ve normalizasyon tamamlandı.")

# --- 3. Hibrit Yapı İçin Gerekli Modellerin Eğitimi ---
print("\n" + "=" * 50 + "\n--- 3. Hibrit Yapı İçin Gerekli Modellerin Eğitimi ---\n" + "=" * 50)

# Stacking modelinin tanımı
base_estimators = [('xgb', XGBRegressor(random_state=42)), ('huber', HuberRegressor())]
stacking_model = StackingRegressor(estimators=base_estimators,
                                   final_estimator=GradientBoostingRegressor(random_state=42))

# Eğitilecek tüm temel modeller
models_to_train = {
    'XGBoost': XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=400, max_depth=7, learning_rate=0.05, random_state=42, n_jobs=-1),
    'Stacking_Ensemble': stacking_model
}
trained_models = {}

# İnce ayar yapılacak hedefler ve parametre aralıkları
tuning_map = {
    'rel_alt_akma_dayanimi': {
        'model': models_to_train['LightGBM'],
        'params': {'n_estimators': [300, 400], 'max_depth': [6, 8], 'learning_rate': [0.05, 0.07]}
    },
    'reh_ust_akma_dayanimi': {
        'model': models_to_train['Stacking_Ensemble'],
        'params': {'xgb__n_estimators': [100, 200], 'final_estimator__n_estimators': [50, 100]}
    }
}

# Tüm model ailelerini ve hedefleri eğit
for name, model in models_to_train.items():
    print(f"\n--- {name} Modeli Eğitiliyor ---")
    target_specific_models = {}
    for target in target_columns:
        y_train_single_target = y_train_transformed[target]

        # Eğer bu hedef için özel bir optimizasyon varsa, onu yap
        if target in tuning_map and name == models_to_train[name].__class__.__name__:
            print(f"'{target}' için GridSearchCV ile ince ayar yapılıyor...")
            search = GridSearchCV(model, param_grid=tuning_map[target]['params'], cv=3, scoring='r2', n_jobs=-1)
            search.fit(X_train_scaled, y_train_single_target)
            best_model_for_target = search.best_estimator_
        else:  # Optimizasyon yoksa, standart parametrelerle eğit
            best_model_for_target = clone(model)
            best_model_for_target.fit(X_train_scaled, y_train_single_target)

        target_specific_models[target] = best_model_for_target
    trained_models[name] = target_specific_models

joblib.dump(trained_models, 'model_portfolio_hybrid.pkl')
print("\nTüm hibrit model bileşenleri eğitildi ve 'model_portfolio_hybrid.pkl' olarak kaydedildi.")

# --- 4. Hibrit Model ile Nihai Tahmin ---
print("\n" + "=" * 50 + "\n--- 4. Yeni Veri ile Nihai Hibrit Tahmin ---\n" + "=" * 50)
try:
    model_portfolio = joblib.load('model_portfolio_hybrid.pkl')
    scaler = joblib.load('minmax_scaler.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    training_medians = joblib.load('training_medians.joblib')

    new_data_raw = pd.read_csv('yeni_dokum_veri.csv', sep=';', decimal=',')
    print("'yeni_dokum_veri.csv' dosyası başarıyla okundu.")

    new_data_raw = new_data_raw.loc[:, ~new_data_raw.columns.duplicated()]
    new_data_processed = preprocess_data(new_data_raw, base_columns_info=feature_columns, medians=training_medians)
    new_data_aligned = new_data_processed[feature_columns]
    new_data_scaled = scaler.transform(new_data_aligned)

    # Nihai hibrit model haritası
    hybrid_map = {
        'cap': 'XGBoost',
        'ovalite': 'XGBoost',
        'tufal_orani': 'GradientBoosting',
        'reh_ust_akma_dayanimi': 'Stacking_Ensemble',
        'elastikiyet': 'Ensemble_of_Ensembles',  # Özel Komite
        'rel_alt_akma_dayanimi': 'LightGBM'
    }

    print("\n--- Kullanılan Hibrit Model Haritası ---")
    print(hybrid_map)

    final_predictions = {}

    # Her hedef için ilgili uzman modeli kullanarak tahmin yap
    for target, model_name in hybrid_map.items():
        if model_name == 'Ensemble_of_Ensembles':
            # Elastikiyet için XGBoost ve Stacking'in tahminlerinin ortalamasını al
            xgb_model = model_portfolio['XGBoost'][target]
            stacking_model = model_portfolio['Stacking_Ensemble'][target]

            pred_xgb_transformed = xgb_model.predict(new_data_scaled)
            pred_stacking_transformed = stacking_model.predict(new_data_scaled)

            # İki modelin tahminlerinin ortalamasını alarak daha stabil bir sonuç elde et
            avg_pred_transformed = (pred_xgb_transformed + pred_stacking_transformed) / 2
            final_predictions[f'pred_{target}'] = np.expm1(avg_pred_transformed)
        else:
            model = model_portfolio[model_name][target]
            pred_transformed = model.predict(new_data_scaled)
            final_predictions[f'pred_{target}'] = np.expm1(pred_transformed)

    final_predictions_df = pd.DataFrame(final_predictions)

    print("\n>>> NİHAİ HİBRİT MODEL İLE YENİ DÖKÜM TAHMİNLERİ (İlk 10):")
    # Sütunları orijinal sıraya göre göster
    print(final_predictions_df[[f'pred_{col}' for col in target_columns]].head(10).round(3))

    output_csv_path = 'yeni_dokum_tahmin_sonuclari_hibrit_final.csv'
    final_predictions_df.to_csv(output_csv_path, index=False)
    print(f"\nSon tahmin sonuçları '{output_csv_path}' dosyasına kaydedildi.")

except FileNotFoundError:
    print("\nUYARI: 'yeni_dokum_veri.csv' bulunamadı.")
except Exception as e:
    print(f"\nYeni veri ile tahmin sırasında hata: {e}")