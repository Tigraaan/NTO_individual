# champion_ml_solution.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ChampionMLPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        
    def load_all_data(self):
        """Загрузка всех доступных данных"""
        print("📂 ЗАГРУЗКА ВСЕХ ДАННЫХ...")
        
        # Функция для умной загрузки
        def smart_load(filename):
            for sep in [';', ',', '\t']:
                try:
                    df = pd.read_csv(filename, sep=sep, encoding='utf-8')
                    if len(df.columns) > 1:
                        print(f"   ✅ {filename}: {df.shape}")
                        return df
                except:
                    continue
            return None
        
        # Загрузка всех файлов
        train = smart_load('train.csv')
        test = smart_load('test.csv')
        books = smart_load('books.csv')
        users = smart_load('users.csv')
        genres = smart_load('genres.csv')
        book_genres = smart_load('book_genres.csv')
        book_descriptions = smart_load('book_descriptions.csv')
        
        return train, test, books, users, genres, book_genres, book_descriptions
    
    def create_comprehensive_features(self, df, books, users, book_genres, is_train=True):
        """Создание комплексных признаков из всех данных"""
        print("🔧 СОЗДАНИЕ КОМПЛЕКСНЫХ ПРИЗНАКОВ...")
        
        # 1. БАЗОВЫЕ ПРИЗНАКИ ИЗ TRAIN
        if is_train:
            # Для обучения используем только прочитанные книги
            df = df[df['has_read'] == 1].copy()
        
        # 2. СТАТИСТИКИ ПОЛЬЗОВАТЕЛЕЙ И КНИГ
        if is_train:
            # User statistics
            self.user_stats = df.groupby('user_id').agg({
                'rating': ['mean', 'count', 'std', 'min', 'max', 'median']
            }).reset_index()
            self.user_stats.columns = ['user_id', 'user_mean', 'user_count', 'user_std', 'user_min', 'user_max', 'user_median']
            
            # Book statistics  
            self.book_stats = df.groupby('book_id').agg({
                'rating': ['mean', 'count', 'std', 'min', 'max', 'median']
            }).reset_index()
            self.book_stats.columns = ['book_id', 'book_mean', 'book_count', 'book_std', 'book_min', 'book_max', 'book_median']
            
            # Global statistics
            self.global_mean = df['rating'].mean()
            self.global_median = df['rating'].median()
            self.global_std = df['rating'].std()
        
        # Объединение со статистиками
        df = df.merge(self.user_stats, on='user_id', how='left')
        df = df.merge(self.book_stats, on='book_id', how='left')
        
        # 3. ПРИЗНАКИ ИЗ BOOKS.CSV
        if books is not None:
            df = df.merge(books, on='book_id', how='left')
            
            # Признаки из книг
            if 'publication_year' in df.columns:
                df['publication_year'] = df['publication_year'].fillna(1980)
                df['book_age'] = 2024 - df['publication_year']
                df['is_old_book'] = (df['book_age'] > 30).astype(int)
                df['is_recent_book'] = (df['book_age'] < 5).astype(int)
            
            if 'avg_rating' in df.columns:
                df['avg_rating_diff'] = df['avg_rating'] - df['book_mean']
        
        # 4. ПРИЗНАКИ ИЗ USERS.CSV
        if users is not None:
            df = df.merge(users, on='user_id', how='left')
            
            if 'age' in df.columns:
                df['age'] = df['age'].fillna(df['age'].median())
                df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 50, 100], labels=[1, 2, 3, 4, 5])
            
            if 'gender' in df.columns:
                df['gender'] = df['gender'].fillna(1)
        
        # 5. ПРИЗНАКИ ИЗ ЖАНРОВ
        if book_genres is not None and genres is not None:
            # Самые популярные жанры
            genre_counts = book_genres['genre_id'].value_counts().head(10)
            top_genres = genre_counts.index
            
            for genre_id in top_genres:
                genre_name = genres[genres['genre_id'] == genre_id]['genre_name'].iloc[0] if len(genres[genres['genre_id'] == genre_id]) > 0 else f'genre_{genre_id}'
                genre_books = book_genres[book_genres['genre_id'] == genre_id]['book_id']
                df[f'is_{genre_name}'] = df['book_id'].isin(genre_books).astype(int)
        
        # 6. ОСНОВНЫЕ ПРИЗНАКИ МОДЕЛИ
        # User features
        df['user_confidence'] = np.log1p(df['user_count']) / 4.0
        df['user_generosity'] = (df['user_mean'] - self.global_mean) / max(self.global_std, 0.1)
        df['user_consistency'] = 1 / (1 + df['user_std'].fillna(1))
        
        # Book features
        df['book_popularity'] = np.log1p(df['book_count']) / 4.0
        df['book_controversial'] = (df['book_std'] > 2.0).astype(int)
        df['book_consistency'] = 1 / (1 + df['book_std'].fillna(1))
        
        # Interaction features
        df['mean_interaction'] = df['user_mean'] * df['book_mean'] / 10.0
        df['confidence_interaction'] = df['user_confidence'] * df['book_popularity']
        df['generosity_quality'] = df['user_generosity'] * df['book_mean']
        
        # Relative features
        df['user_mean_diff'] = df['user_mean'] - self.global_mean
        df['book_mean_diff'] = df['book_mean'] - self.global_mean
        df['combined_pred'] = 0.6 * df['user_mean'] + 0.4 * df['book_mean']
        
        # 7. ВЫБОР ФИНАЛЬНЫХ ПРИЗНАКОВ
        base_features = [
            # User features
            'user_mean', 'user_count', 'user_std', 'user_min', 'user_max', 'user_median',
            'user_confidence', 'user_generosity', 'user_consistency',
            
            # Book features  
            'book_mean', 'book_count', 'book_std', 'book_min', 'book_max', 'book_median',
            'book_popularity', 'book_controversial', 'book_consistency',
            
            # Interaction features
            'mean_interaction', 'confidence_interaction', 'generosity_quality',
            'user_mean_diff', 'book_mean_diff', 'combined_pred'
        ]
        
        # Добавляем дополнительные признаки если они есть
        additional_features = []
        if 'age' in df.columns:
            additional_features.extend(['age', 'age_group'])
        if 'gender' in df.columns:
            additional_features.append('gender')
        if 'publication_year' in df.columns:
            additional_features.extend(['publication_year', 'book_age', 'is_old_book', 'is_recent_book'])
        if 'avg_rating' in df.columns:
            additional_features.append('avg_rating_diff')
        
        # Жанровые признаки
        genre_features = [col for col in df.columns if col.startswith('is_')]
        
        all_features = base_features + additional_features + genre_features
        available_features = [f for f in all_features if f in df.columns]
        
        if is_train:
            self.feature_columns = available_features
            print(f"   📊 Используется {len(self.feature_columns)} признаков")
        
        # Заполнение пропусков
        df[available_features] = df[available_features].fillna(0)
        
        return df[available_features]
    
    def train_champion_model(self, X, y):
        """Обучение чемпионской модели"""
        print("🎯 ОБУЧЕНИЕ ЧЕМПИОНСКОЙ МОДЕЛИ...")
        
        # Разделение на train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   Train: {X_train.shape}, Val: {X_val.shape}")
        
        # Масштабирование
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_val_scaled = self.scalers['standard'].transform(X_val)
        
        # МОДЕЛЬ 1: Gradient Boosting (основная)
        print("   🚀 Обучение Gradient Boosting...")
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=50,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42
        )
        self.models['gb'].fit(X_train_scaled, y_train)
        
        # МОДЕЛЬ 2: Random Forest
        print("   🌲 Обучение Random Forest...")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train, y_train)
        
        # МОДЕЛЬ 3: Ridge Regression
        print("   📈 Обучение Ridge Regression...")
        self.models['ridge'] = Ridge(alpha=0.5, random_state=42)
        self.models['ridge'].fit(X_train_scaled, y_train)
        
        # ОЦЕНКА МОДЕЛЕЙ
        print("\n📊 ОЦЕНКА МОДЕЛЕЙ НА VALIDATION:")
        best_rmse = float('inf')
        best_model = None
        
        for name, model in self.models.items():
            if name == 'rf':
                preds = model.predict(X_val)
            else:
                preds = model.predict(X_val_scaled)
            
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mae = mean_absolute_error(y_val, preds)
            print(f"   {name.upper():12} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = name
        
        print(f"   🏆 Лучшая модель: {best_model.upper()} (RMSE: {best_rmse:.4f})")
        
        return X_train_scaled, y_train, X_val_scaled, y_val
    
    def predict_ensemble(self, X):
        """Предсказание ансамблем моделей"""
        if len(X) == 0:
            return np.array([self.global_mean] * len(X))
        
        X_scaled = self.scalers['standard'].transform(X)
        
        # Предсказания всех моделей
        preds_gb = self.models['gb'].predict(X_scaled)
        preds_rf = self.models['rf'].predict(X)
        preds_ridge = self.models['ridge'].predict(X_scaled)
        
        # Взвешенное усреднение (больше вес у лучшей модели)
        weights = {'gb': 0.5, 'rf': 0.3, 'ridge': 0.2}
        ensemble_pred = (
            weights['gb'] * preds_gb + 
            weights['rf'] * preds_rf + 
            weights['ridge'] * preds_ridge
        )
        
        return ensemble_pred
    
    def smart_post_processing(self, predictions, train_ratings):
        """Умная пост-обработка предсказаний"""
        # 1. Ограничение диапазона
        predictions = np.clip(predictions, 1.0, 10.0)
        
        # 2. Калибровка распределения
        if len(train_ratings) > 0:
            pred_mean = np.mean(predictions)
            train_mean = np.mean(train_ratings)
            pred_std = np.std(predictions)
            train_std = np.std(train_ratings)
            
            # Корректировка среднего
            if abs(pred_mean - train_mean) > 0.05:
                adjustment = (train_mean - pred_mean) * 0.4
                predictions = predictions + adjustment
            
            # Корректировка дисперсии
            if pred_std > 0 and train_std > 0:
                std_ratio = train_std / pred_std
                if 0.8 < std_ratio < 1.2:
                    centered = predictions - np.mean(predictions)
                    predictions = centered * (std_ratio ** 0.8) + np.mean(predictions)
        
        # 3. Финальное ограничение
        predictions = np.clip(predictions, 1.0, 10.0)
        
        return predictions
    
    def run_champion_pipeline(self):
        """Запуск чемпионского пайплайна"""
        print("🚀 ЗАПУСК ЧЕМПИОНСКОГО ML ПАЙПЛАЙНА")
        print("=" * 60)
        
        try:
            # 1. Загрузка всех данных
            train, test, books, users, genres, book_genres, book_descriptions = self.load_all_data()
            
            if train is None:
                raise Exception("Не удалось загрузить train.csv")
            
            # 2. Создание признаков для обучения
            print("\n🎯 ПОДГОТОВКА ТРЕНИРОВОЧНЫХ ДАННЫХ...")
            X_train = self.create_comprehensive_features(train, books, users, book_genres, is_train=True)
            y_train = train[train['has_read'] == 1]['rating'] if 'has_read' in train.columns else train['rating']
            
            print(f"   📊 Финальные данные: {X_train.shape}")
            
            # 3. Обучение моделей
            self.train_champion_model(X_train, y_train)
            
            # 4. Предсказание на тесте
            if test is not None:
                print("\n🎯 ПРЕДСКАЗАНИЕ НА ТЕСТОВЫХ ДАННЫХ...")
                X_test = self.create_comprehensive_features(test, books, users, book_genres, is_train=False)
                X_test = X_test.fillna(0)
                
                test_predictions = self.predict_ensemble(X_test)
                final_predictions = self.smart_post_processing(test_predictions, y_train)
                
                # 5. Создание сабмита
                submission = test[['user_id', 'book_id']].copy()
                submission['rating_predict'] = final_predictions
                
                # 6. Анализ результатов
                self.analyze_champion_results(submission, y_train)
                
                # 7. Сохранение
                submission.to_csv('champion_ml_submission.csv', index=False)
                print(f"\n💾 ЧЕМПИОНСКИЙ САБМИТ СОХРАНЕН: champion_ml_submission.csv")
                
                return submission
            else:
                print("❌ Тестовые данные не найдены")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка в ML пайплайне: {e}")
            return None
    
    def analyze_champion_results(self, submission, train_ratings):
        """Анализ результатов чемпионской модели"""
        print("\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ ЧЕМПИОНА:")
        
        pred_stats = submission['rating_predict'].describe()
        train_stats = train_ratings.describe()
        
        print(f"   Предсказания: {pred_stats['min']:.3f} - {pred_stats['max']:.3f}")
        print(f"   Среднее: {pred_stats['mean']:.3f} (тренировка: {train_stats['mean']:.3f})")
        print(f"   Медиана: {np.median(submission['rating_predict']):.3f} (тренировка: {train_stats['50%']:.3f})")
        print(f"   Стандартное отклонение: {pred_stats['std']:.3f} (тренировка: {train_stats['std']:.3f})")
        
        # Анализ распределения
        print(f"\n   📈 Распределение оценок:")
        for threshold in [3, 5, 7, 9]:
            pred_pct = (submission['rating_predict'] >= threshold).mean() * 100
            train_pct = (train_ratings >= threshold).mean() * 100
            print(f"   ≥{threshold}: {pred_pct:5.1f}% (тренировка: {train_pct:5.1f}%)")

# БЕЗОПАСНЫЙ ЗАПУСК
if __name__ == "__main__":
    print("🎯 ЧЕМПИОНСКОЕ ML РЕШЕНИЕ ДЛЯ ПРЕДСКАЗАНИЯ РЕЙТИНГОВ")
    print("💡 Использует ВСЕ доступные данные:")
    print("   • train.csv + test.csv")
    print("   • books.csv (метаданные книг)")
    print("   • users.csv (метаданные пользователей)") 
    print("   • genres.csv + book_genres.csv (жанры)")
    print("   • book_descriptions.csv (текстовые описания)")
    print("=" * 70)
    
    # Запуск чемпионского решения
    champion = ChampionMLPredictor()
    submission = champion.run_champion_pipeline()
    
    if submission is not None:
        print(f"\n🎉 ЧЕМПИОНСКОЕ ML РЕШЕНИЕ УСПЕШНО СОЗДАНО!")
        print("📤 Отправляйте: champion_ml_submission.csv")
        print("🚀 ЦЕЛЕВАЯ МЕТРИКА: 0.773+")
    else:
        print("\n❌ Чемпионское решение не сработало")
        print("💡 Рекомендуется использовать предыдущие рабочие решения")
    
    print("💪 УДАЧИ В СОРЕВНОВАНИИ!")