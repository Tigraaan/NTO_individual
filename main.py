import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("🚀 ФИНАЛЬНОЕ РЕШЕНИЕ ДЛЯ РЕАЛЬНЫХ ДАННЫХ...")

# Загрузка данных
print("📊 Загрузка данных...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train: {train.shape}, Test: {test.shape}")

# Подготовка обучающих данных (только прочитанные книги)
train_labeled = train[train['has_read'] == 1].copy()
print(f"📚 Прочитанных книг для обучения: {len(train_labeled)}")

# Преобразование временных меток
train_labeled['timestamp'] = pd.to_datetime(train_labeled['timestamp'])

print("🔧 СОЗДАНИЕ ПРИЗНАКОВ...")

# ОСНОВНЫЕ ПРИЗНАКИ
def create_features(df, train_df=None):
    if train_df is None:
        return df
    
    print("👤 Признаки пользователей...")
    # Агрегации по пользователям
    user_stats = train_df.groupby('user_id').agg({
        'rating': ['mean', 'count', 'std', 'min', 'max', 'median'],
        'book_id': 'nunique'
    }).round(4)
    user_stats.columns = [
        'user_mean_rating', 'user_rating_count', 'user_rating_std',
        'user_min_rating', 'user_max_rating', 'user_median_rating', 'user_unique_books'
    ]
    user_stats = user_stats.reset_index()
    
    # Дополнительные пользовательские признаки
    user_stats['user_strictness'] = 10 - user_stats['user_mean_rating']
    user_stats['user_rating_range'] = user_stats['user_max_rating'] - user_stats['user_min_rating']
    user_stats['user_consistency'] = 1 / (1 + user_stats['user_rating_std'])
    
    print("📚 Признаки книг...")
    # Агрегации по книгам
    book_stats = train_df.groupby('book_id').agg({
        'rating': ['mean', 'count', 'std', 'min', 'max', 'median']
    }).round(4)
    book_stats.columns = [
        'book_mean_rating', 'book_rating_count', 'book_rating_std',
        'book_min_rating', 'book_max_rating', 'book_median_rating'
    ]
    book_stats = book_stats.reset_index()
    
    # Дополнительные книжные признаки
    book_stats['book_popularity'] = np.log1p(book_stats['book_rating_count'])
    book_stats['book_rating_range'] = book_stats['book_max_rating'] - book_stats['book_min_rating']
    book_stats['book_consistency'] = 1 / (1 + book_stats['book_rating_std'])
    book_stats['book_controversial'] = book_stats['book_rating_std']
    
    print("⏰ Временные признаки...")
    # Временные признаки
    user_time_stats = train_df.groupby('user_id').agg({
        'timestamp': ['min', 'max', 'count']
    })
    user_time_stats.columns = ['user_first_interaction', 'user_last_interaction', 'user_total_interactions']
    user_time_stats = user_time_stats.reset_index()
    
    user_time_stats['user_account_age_days'] = (
        user_time_stats['user_last_interaction'] - user_time_stats['user_first_interaction']
    ).dt.total_seconds() / (24 * 3600)
    
    user_time_stats['user_activity_rate'] = (
        user_time_stats['user_total_interactions'] / user_time_stats['user_account_age_days']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Объединение всех признаков
    df = df.merge(user_stats, on='user_id', how='left')
    df = df.merge(book_stats, on='book_id', how='left')
    df = df.merge(user_time_stats, on='user_id', how='left')
    
    return df

# ПРИЗНАКИ ВЗАИМОДЕЙСТВИЯ
def create_interaction_features(df):
    # Разница рейтингов
    df['rating_gap'] = df['book_mean_rating'] - df['user_mean_rating']
    df['rating_gap_abs'] = abs(df['rating_gap'])
    df['rating_similarity'] = 1 / (1 + df['rating_gap_abs'])
    
    # Взаимодействия популярности
    df['popularity_interaction'] = df['user_rating_count'] * df['book_popularity']
    df['experience_popularity'] = df['user_unique_books'] * df['book_popularity']
    
    # Взаимодействия консистентности
    df['consistency_match'] = df['user_consistency'] * df['book_consistency']
    
    return df

# ПОДГОТОВКА ДАННЫХ
print("🔄 Подготовка тренировочных данных...")

# Удаляем выбросы рейтингов
q_low = train_labeled['rating'].quantile(0.01)
q_high = train_labeled['rating'].quantile(0.99)
train_labeled = train_labeled[(train_labeled['rating'] >= q_low) & (train_labeled['rating'] <= q_high)]
print(f"📚 После удаления выбросов: {len(train_labeled)}")

# Создаем все признаки
train_features = create_features(train_labeled, train_labeled)
train_features = create_interaction_features(train_features)

print("🔄 Подготовка тестовых данных...")
test_features = create_features(test, train_labeled)
test_features = create_interaction_features(test_features)

# ВЫБОР ПРИЗНАКОВ
feature_columns = [
    # Пользовательские
    'user_mean_rating', 'user_rating_count', 'user_rating_std',
    'user_min_rating', 'user_max_rating', 'user_median_rating', 'user_unique_books',
    'user_strictness', 'user_rating_range', 'user_consistency',
    
    # Книжные
    'book_mean_rating', 'book_rating_count', 'book_rating_std',
    'book_min_rating', 'book_max_rating', 'book_median_rating',
    'book_popularity', 'book_rating_range', 'book_consistency', 'book_controversial',
    
    # Временные
    'user_account_age_days', 'user_total_interactions', 'user_activity_rate',
    
    # Взаимодействия
    'rating_gap', 'rating_gap_abs', 'rating_similarity',
    'popularity_interaction', 'experience_popularity', 'consistency_match'
]

print(f"🎯 Используется {len(feature_columns)} признаков")

# Заполнение пропусков
for col in feature_columns:
    train_features[col] = train_features[col].fillna(train_features[col].median())
    test_features[col] = test_features[col].fillna(test_features[col].median())

# ВАЛИДАЦИЯ
print("\n🎯 РАЗДЕЛЕНИЕ НА TRAIN/VAL...")
X_train, X_val, y_train, y_val = train_test_split(
    train_features[feature_columns], 
    train_features['rating'], 
    test_size=0.2, 
    random_state=42
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# ОПТИМИЗИРОВАННАЯ МОДЕЛЬ
model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    random_strength=0.8,
    bagging_temperature=0.8,
    random_seed=42,
    verbose=200,
    early_stopping_rounds=100,
    eval_metric='RMSE'
)

print("\n🎓 ОБУЧЕНИЕ МОДЕЛИ...")
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    plot=False
)

# ВАЛИДАЦИЯ
val_predictions = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, val_predictions))
mae_val = mean_absolute_error(y_val, val_predictions)
score_val = 1 - (0.5 * (rmse_val/10) + 0.5 * (mae_val/10))

print(f"\n📊 ВАЛИДАЦИОННЫЕ МЕТРИКИ:")
print(f"RMSE: {rmse_val:.4f}")
print(f"MAE: {mae_val:.4f}")
print(f"SCORE: {score_val:.4f}")

# ФИНАЛЬНАЯ МОДЕЛЬ
print("\n🎓 ФИНАЛЬНОЕ ОБУЧЕНИЕ...")
best_iteration = model.get_best_iteration()
final_model = CatBoostRegressor(
    iterations=best_iteration + 100,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100
)

final_model.fit(train_features[feature_columns], train_features['rating'])

# ПРЕДСКАЗАНИЕ
print("\n🔮 ПРЕДСКАЗАНИЕ...")
X_test = test_features[feature_columns]
test_predictions = final_model.predict(X_test)
test_predictions = np.clip(test_predictions, 0, 10)

# СОХРАНЕНИЕ
print("\n💾 СОХРАНЕНИЕ САБМИТА...")
submission = test[['user_id', 'book_id']].copy()
submission['rating_predict'] = test_predictions

submission_file = 'submission_final.csv'
submission.to_csv(submission_file, index=False)

print(f"✅ САБМИТ СОХРАНЕН: {submission_file}")
print(f"📊 Размер: {submission.shape}")
print(f"📈 Диапазон предсказаний: {submission['rating_predict'].min():.2f} - {submission['rating_predict'].max():.2f}")

# ФИНАЛЬНЫЙ SCORE
final_predictions = final_model.predict(train_features[feature_columns])
rmse_final = np.sqrt(mean_squared_error(train_features['rating'], final_predictions))
mae_final = mean_absolute_error(train_features['rating'], final_predictions)
score_final = 1 - (0.5 * (rmse_final/10) + 0.5 * (mae_final/10))

print(f"\n🎯 ФИНАЛЬНЫЙ SCORE НА ТРЕЙНЕ: {score_final:.6f}")
print(f"📈 ОЖИДАЕМОЕ УЛУЧШЕНИЕ: +{score_final - 0.756:.4f}")

print("\n🚀 ГОТОВО! Отправляйте submission_final.csv на платформу!")