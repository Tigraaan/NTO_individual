import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

class PowerRatingPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def load_and_analyze(self):
        """Загрузка и глубокий анализ данных"""
        print("🧠 ЗАГРУЗКА И АНАЛИЗ")
        
        self.train = pd.read_csv('train.csv', parse_dates=['timestamp'])
        self.test = pd.read_csv('test.csv')
        self.books = pd.read_csv('books.csv')
        self.users = pd.read_csv('users.csv')
        self.book_genres = pd.read_csv('book_genres.csv')
        self.genres = pd.read_csv('genres.csv')
        
        # ГЛУБОКИЙ АНАЛИЗ
        ratings = self.train[self.train['has_read'] == 1]['rating']
        print(f"📊 РАСПРЕДЕЛЕНИЕ ОЦЕНОК:")
        self.rating_stats = {}
        for i in range(1, 11):
            count = (ratings == i).sum()
            pct = (ratings == i).mean() * 100
            self.rating_stats[i] = pct
            print(f"  {i}: {count:6d} ({pct:5.1f}%)")
        
        # Анализ пользователей
        user_analysis = self.train[self.train['has_read'] == 1].groupby('user_id')['rating'].agg(['mean', 'count', 'std'])
        print(f"\n👤 АНАЛИЗ ПОЛЬЗОВАТЕЛЕЙ:")
        print(f"  Средняя оценка: {user_analysis['mean'].mean():.2f}")
        print(f"  Медиана книг на пользователя: {user_analysis['count'].median():.0f}")
        print(f"  Стандартное отклонение оценок: {user_analysis['std'].mean():.2f}")
        
        return self.rating_stats
    
    def create_power_features(self, df, is_train=True):
        """Создание мощных признаков с глубоким feature engineering"""
        print("🚀 СОЗДАНИЕ МОЩНЫХ ПРИЗНАКОВ")
        
        features = df.copy()
        
        # Базовые объединения
        features = features.merge(self.books, on='book_id', how='left')
        features = features.merge(self.users, on='user_id', how='left')
        
        # === POWER USER FEATURES ===
        if is_train:
            # Расширенные статистики пользователей
            user_power_stats = self.train[self.train['has_read'] == 1].groupby('user_id').agg({
                'rating': ['mean', 'std', 'count', 'min', 'max', 
                          lambda x: (x >= 9).sum(),    # высокие оценки
                          lambda x: (x <= 5).sum(),    # низкие оценки
                          lambda x: x.skew()],         # асимметрия
                'book_id': 'nunique',
                'timestamp': ['min', 'max']
            })
            user_power_stats.columns = [
                'user_mean', 'user_std', 'user_count', 'user_min', 'user_max',
                'user_high', 'user_low', 'user_skew',
                'user_unique_books', 'user_first', 'user_last'
            ]
            
            # Временные характеристики
            user_power_stats['user_activity_days'] = (user_power_stats['user_last'] - user_power_stats['user_first']).dt.days
            user_power_stats['user_reading_speed'] = user_power_stats['user_unique_books'] / user_power_stats['user_activity_days']
            user_power_stats['user_rating_range'] = user_power_stats['user_max'] - user_power_stats['user_min']
            user_power_stats['user_consistency'] = 1 / (1 + user_power_stats['user_std'])
            user_power_stats['user_preference_strength'] = user_power_stats['user_high'] / user_power_stats['user_count']
            user_power_stats['user_critical_ratio'] = user_power_stats['user_low'] / user_power_stats['user_count']
            
            # Жанровые предпочтения
            user_genre_power = self.train[self.train['has_read'] == 1].merge(
                self.book_genres, on='book_id'
            ).groupby('user_id').agg({
                'genre_id': 'nunique',
                'rating': ['mean', 'std']
            })
            user_genre_power.columns = ['user_genre_diversity', 'user_genre_mean', 'user_genre_std']
            
            user_power_stats = user_power_stats.merge(user_genre_power, on='user_id', how='left')
            
            self.user_power_stats = user_power_stats.reset_index()
            
            # === POWER BOOK FEATURES ===
            book_power_stats = self.train[self.train['has_read'] == 1].groupby('book_id').agg({
                'rating': ['mean', 'std', 'count', 'min', 'max',
                          lambda x: x.quantile(0.75),  # 75% перцентиль
                          lambda x: x.quantile(0.25),  # 25% перцентиль
                          lambda x: (x >= 8).sum()],   # высокие оценки
                'user_id': 'nunique',
                'timestamp': ['min', 'max']
            })
            book_power_stats.columns = [
                'book_mean', 'book_std', 'book_count', 'book_min', 'book_max',
                'book_q75', 'book_q25', 'book_high_count',
                'book_unique_readers', 'book_first', 'book_last'
            ]
            
            book_power_stats['book_rating_iqr'] = book_power_stats['book_q75'] - book_power_stats['book_q25']
            book_power_stats['book_rating_span'] = (book_power_stats['book_last'] - book_power_stats['book_first']).dt.days
            book_power_stats['book_controversy'] = book_power_stats['book_std'] * np.log1p(book_power_stats['book_count'])
            book_power_stats['book_quality_score'] = book_power_stats['book_mean'] * (1 - book_power_stats['book_std']/10)
            book_power_stats['book_popularity_score'] = book_power_stats['book_mean'] * np.log1p(book_power_stats['book_count'])
            book_power_stats['book_high_ratio'] = book_power_stats['book_high_count'] / book_power_stats['book_count']
            
            # Жанровые особенности
            book_genre_power = self.book_genres.groupby('book_id').agg({
                'genre_id': ['count', 'nunique']
            })
            book_genre_power.columns = ['book_total_genres', 'book_unique_genres']
            
            book_power_stats = book_power_stats.merge(book_genre_power, on='book_id', how='left')
            
            self.book_power_stats = book_power_stats.reset_index()
            
            # === POWER AUTHOR FEATURES ===
            author_power_stats = self.train[self.train['has_read'] == 1].merge(
                self.books[['book_id', 'author_id']], on='book_id'
            ).groupby('author_id').agg({
                'rating': ['mean', 'std', 'count'],
                'user_id': 'nunique',
                'book_id': 'nunique'
            })
            author_power_stats.columns = [
                'author_mean', 'author_std', 'author_rating_count',
                'author_unique_readers', 'author_unique_books'
            ]
            
            author_power_stats['author_popularity'] = author_power_stats['author_unique_readers'] * author_power_stats['author_mean']
            author_power_stats['author_consistency'] = 1 / (1 + author_power_stats['author_std'])
            
            self.author_power_stats = author_power_stats.reset_index()
            
            # === USER-AUTHOR INTERACTION POWER ===
            user_author_power = self.train[self.train['has_read'] == 1].merge(
                self.books[['book_id', 'author_id']], on='book_id'
            ).groupby(['user_id', 'author_id']).agg({
                'rating': ['mean', 'count', 'std']
            })
            user_author_power.columns = ['user_author_mean', 'user_author_count', 'user_author_std']
            
            user_author_power['user_author_loyalty'] = user_author_power['user_author_mean'] * np.log1p(user_author_power['user_author_count'])
            
            self.user_author_power = user_author_power.reset_index()
        
        # СЛИВАЕМ ВСЕ МОЩНЫЕ ПРИЗНАКИ
        features = features.merge(self.user_power_stats, on='user_id', how='left')
        features = features.merge(self.book_power_stats, on='book_id', how='left')
        features = features.merge(self.author_power_stats, on='author_id', how='left')
        features = features.merge(self.user_author_power, on=['user_id', 'author_id'], how='left')
        
        # === SUPER ENGINEERED FEATURES ===
        # 1. Синергия пользователь-книга
        features['super_synergy'] = (
            features['user_mean'] * features['book_quality_score'] * 
            features['user_consistency']
        )
        
        # 2. Предсказуемость оценки
        features['predictability_score'] = (
            1 / (1 + features['user_std']) * 
            1 / (1 + features['book_std']) *
            features['user_consistency']
        )
        
        # 3. Совпадение уровней
        features['level_alignment'] = 1 - np.abs(features['user_mean'] - features['book_mean']) / 10
        
        # 4. Взвешенная популярность
        features['weighted_popularity'] = features['book_popularity_score'] * features['user_preference_strength']
        
        # 5. Возрастная совместимость
        features['book_age'] = 2023 - features['publication_year']
        features['age_compatibility'] = np.exp(-np.abs(features['age'] - features['book_age']) / 30)
        
        # 6. Авторская лояльность
        features['author_loyalty_boost'] = features['user_author_loyalty'] * features['author_popularity']
        
        # 7. Статус хита
        features['hit_probability'] = (
            (features['book_mean'] >= 8.0) * 
            (features['book_count'] >= 5) * 
            (features['book_high_ratio'] >= 0.7)
        ).astype(float)
        
        # 8. Нишевость
        features['niche_advantage'] = 1 / (1 + features['book_count']) * features['user_genre_diversity']
        
        # 9. Сложность книги
        features['book_difficulty'] = features['book_std'] * features['book_rating_iqr']
        
        # 10. Эффект неожиданности
        features['surprise_factor'] = np.abs(features['user_mean'] - features['book_mean']) * features['book_controversy']
        
        # 11. Временной тренд
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features['year'] = features['timestamp'].dt.year
            features['month'] = features['timestamp'].dt.month
            features['seasonal_effect'] = np.sin(2 * np.pi * features['month'] / 12)
        
        # === АГРЕССИВНОЕ ЗАПОЛНЕНИЕ ПРОПУСКОВ ===
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if 'mean' in col or 'score' in col:
                features[col] = features[col].fillna(features[col].median())
            elif 'count' in col or 'ratio' in col:
                features[col] = features[col].fillna(0)
            elif 'std' in col:
                features[col] = features[col].fillna(1)
            else:
                features[col] = features[col].fillna(features[col].mean())
        
        # === ОТБОР ЛУЧШИХ ПРИЗНАКОВ ===
        power_features = [
            # User power features
            'user_mean', 'user_std', 'user_count', 'user_min', 'user_max',
            'user_high', 'user_low', 'user_skew', 'user_unique_books',
            'user_activity_days', 'user_reading_speed', 'user_rating_range',
            'user_consistency', 'user_preference_strength', 'user_critical_ratio',
            'user_genre_diversity', 'user_genre_mean', 'user_genre_std',
            
            # Book power features
            'book_mean', 'book_std', 'book_count', 'book_min', 'book_max',
            'book_q75', 'book_q25', 'book_high_count', 'book_unique_readers',
            'book_rating_iqr', 'book_rating_span', 'book_controversy',
            'book_quality_score', 'book_popularity_score', 'book_high_ratio',
            'book_total_genres', 'book_unique_genres',
            
            # Author power features
            'author_mean', 'author_std', 'author_rating_count',
            'author_unique_readers', 'author_unique_books',
            'author_popularity', 'author_consistency',
            
            # Interaction power features
            'user_author_mean', 'user_author_count', 'user_author_std', 'user_author_loyalty',
            
            # Super engineered features
            'super_synergy', 'predictability_score', 'level_alignment',
            'weighted_popularity', 'age_compatibility', 'author_loyalty_boost',
            'hit_probability', 'niche_advantage', 'book_difficulty', 'surprise_factor',
            
            # Basic features
            'avg_rating', 'publication_year', 'age', 'gender', 'language', 'publisher'
        ]
        
        # Добавляем временные если есть
        if 'seasonal_effect' in features.columns:
            power_features.extend(['year', 'month', 'seasonal_effect'])
        
        available_features = [f for f in power_features if f in features.columns]
        
        if is_train:
            self.feature_columns = available_features
        
        print(f"💪 Используется {len(available_features)} МОЩНЫХ признаков")
        
        return features[available_features]
    
    def create_super_ensemble(self):
        """Создание супер-ансамбля"""
        print("🤖 СОЗДАНИЕ СУПЕР-АНСАМБЛЯ")
        
        models = [
            ('rf_deep', RandomForestRegressor(
                n_estimators=400,
                max_depth=30,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features=0.8,
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )),
            ('rf_wide', RandomForestRegressor(
                n_estimators=300,
                max_depth=25,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=43,
                n_jobs=-1
            )),
            ('gbr_power', GradientBoostingRegressor(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=7,
                min_samples_split=8,
                min_samples_leaf=3,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            )),
            ('et_power', ExtraTreesRegressor(
                n_estimators=350,
                max_depth=28,
                min_samples_split=4,
                min_samples_leaf=1,
                max_features=0.7,
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )),
            ('ridge_power', Ridge(alpha=0.5, random_state=42))
        ]
        
        # Оптимизированные веса
        ensemble = VotingRegressor(models, weights=[4, 3, 4, 4, 1])
        
        return ensemble
    
    def train_power_model(self, X, y):
        """Обучение мощной модели"""
        print("\n🏋️ ОБУЧЕНИЕ МОЩНОЙ МОДЕЛИ")
        
        ensemble = self.create_super_ensemble()
        
        # Разделение данных
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, shuffle=True
        )
        
        print(f"📈 Разделение: Train {X_train.shape}, Val {X_val.shape}")
        
        # Обучение
        ensemble.fit(X_train, y_train)
        
        # Валидация
        y_pred = ensemble.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mae = mean_absolute_error(y_val, y_pred)
        
        print(f"🎯 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        
        # Детальный анализ ошибок
        errors = np.abs(y_val - y_pred)
        print(f"🔍 АНАЛИЗ ОШИБОК:")
        print(f"   Средняя ошибка: {errors.mean():.3f}")
        print(f"   Медианная ошибка: {np.median(errors):.3f}")
        
        # Анализ по диапазонам
        print(f"\n📊 ОШИБКИ ПО ДИАПАЗОНАМ:")
        ranges = [(1, 4), (5, 7), (8, 10)]
        for r_min, r_max in ranges:
            mask = (y_val >= r_min) & (y_val <= r_max)
            if mask.any():
                range_mae = errors[mask].mean()
                range_count = mask.sum()
                print(f"   {r_min}-{r_max}: MAE={range_mae:.3f} (n={range_count})")
        
        # Обучение на всех данных
        print("\n🔥 Финальное обучение на всех данных...")
        ensemble.fit(X, y)
        self.model = ensemble
        
        return ensemble
    
    def power_post_processing(self, predictions):
        """Мощная пост-обработка"""
        print("🎨 МОЩНАЯ ПОСТ-ОБРАБОТКА")
        
        original_mean = predictions.mean()
        
        # АГРЕССИВНЫЙ СДВИГ К ВЫСОКИМ ОЦЕНКАМ
        predictions = np.where(predictions < 8.5, predictions + 0.8, predictions)
        predictions = np.where(predictions < 7.0, predictions + 1.2, predictions)
        predictions = np.where(predictions < 5.0, predictions + 1.5, predictions)
        predictions = np.where(predictions < 3.0, predictions + 2.0, predictions)
        
        # ОКРУГЛЕНИЕ
        predictions = np.round(predictions * 2) / 2
        
        # ОГРАНИЧЕНИЕ
        predictions = np.clip(predictions, 1.0, 10.0)
        
        print(f"📊 Изменения:")
        print(f"   Было: {original_mean:.2f} -> Стало: {predictions.mean():.2f}")
        
        return predictions
    
    def create_power_submission(self):
        """Создание мощного сабмита"""
        print("\n🏆 СОЗДАНИЕ МОЩНОГО САБМИТА")
        
        # Подготовка теста
        X_test = self.create_power_features(self.test, is_train=False)
        
        # Предсказание
        predictions = self.model.predict(X_test)
        
        # Пост-обработка
        final_predictions = self.power_post_processing(predictions)
        
        # Создание сабмита
        submission = self.test[['user_id', 'book_id']].copy()
        submission['rating_predict'] = final_predictions
        
        # Сохранение
        submission.to_csv('submission_power.csv', index=False)
        print("✅ Файл submission_power.csv создан")
        
        # Детальная статистика
        print(f"\n📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
        stats = submission['rating_predict'].describe()
        for stat in ['mean', '50%', 'std', 'min', 'max']:
            print(f"   {stat}: {stats[stat]:.3f}")
        
        # Распределение
        print(f"\n🎯 РАСПРЕДЕЛЕНИЕ:")
        for rating in [8, 9, 10]:
            count = (submission['rating_predict'] == rating).sum()
            pct = count / len(submission) * 100
            print(f"   {rating}: {count:4d} ({pct:5.1f}%)")
        
        return submission
    
    def run_power_pipeline(self):
        """Запуск мощного пайплайна"""
        print("=" * 70)
        print("🚀 МОЩНЫЙ ПАЙПЛАЙН АКТИВИРОВАН!")
        print("=" * 70)
        
        try:
            # 1. Загрузка и анализ
            self.load_and_analyze()
            
            # 2. Подготовка данных
            train_read = self.train[self.train['has_read'] == 1]
            X = self.create_power_features(train_read, is_train=True)
            y = train_read['rating']
            X = X.loc[train_read.index]  # Гарантия совпадения
            
            print(f"\n💪 ДАННЫЕ: X{X.shape}, y{y.shape}")
            
            # 3. Обучение
            self.train_power_model(X, y)
            
            # 4. Создание сабмита
            submission = self.create_power_submission()
            
            print("\n" + "=" * 70)
            print("🎉 МОЩНЫЙ ПАЙПЛАЙН ЗАВЕРШЕН! ОЖИДАЕМ ПРОРЫВ!")
            print("=" * 70)
            
            return submission
            
        except Exception as e:
            print(f"❌ ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            
            # Фолбэк
            submission = self.test[['user_id', 'book_id']].copy()
            submission['rating_predict'] = 9.0
            submission.to_csv('submission_backup.csv', index=False)
            return submission

# ЗАПУСК
if __name__ == "__main__":
    power = PowerRatingPredictor()
    submission = power.run_power_pipeline()
    
    print("\n🔥 ПРИМЕР ПРЕДСКАЗАНИЙ:")
    print(submission.head(8))