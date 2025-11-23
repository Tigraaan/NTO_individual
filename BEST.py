import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

class WorkingRatingPredictor:
    def __init__(self):
        self.model = None
        
    def load_data(self):
        """Загрузка данных"""
        print("📚 ЗАГРУЗКА ДАННЫХ")
        
        self.train = pd.read_csv('train.csv')
        self.test = pd.read_csv('test.csv')
        self.books = pd.read_csv('books.csv')
        self.users = pd.read_csv('users.csv')
        
        print(f"Train: {len(self.train)}, Test: {len(self.test)}")
        
        # Анализ оценок
        ratings = self.train[self.train['has_read'] == 1]['rating']
        print(f"🎯 РАСПРЕДЕЛЕНИЕ ОЦЕНОК:")
        for i in range(1, 11):
            count = (ratings == i).sum()
            pct = (ratings == i).mean() * 100
            if count > 0:
                print(f"  {i}: {count:6d} ({pct:5.1f}%)")
        
        high_ratio = (ratings >= 8).mean()
        print(f"🚨 КРИТИЧЕСКИЙ ФАКТ: {high_ratio:.1%} оценок >= 8")
        
        return ratings
    
    def create_smart_features(self, df, is_train=True):
        """Создание умных признаков БЕЗ ошибок индексов"""
        features = df.copy()
        
        # Базовые объединения
        features = features.merge(self.books, on='book_id', how='left')
        features = features.merge(self.users, on='user_id', how='left')
        
        if is_train:
            # Создаем признаки из тренировочных данных
            train_read = self.train[self.train['has_read'] == 1]
            
            # User features
            user_stats = train_read.groupby('user_id')['rating'].agg(['mean', 'count']).reset_index()
            user_stats.columns = ['user_id', 'user_mean', 'user_count']
            self.user_stats = user_stats
            
            # Book features  
            book_stats = train_read.groupby('book_id')['rating'].agg(['mean', 'count']).reset_index()
            book_stats.columns = ['book_id', 'book_mean', 'book_count']
            self.book_stats = book_stats
        
        # Добавляем признаки через merge (без проблем с индексами)
        features = features.merge(self.user_stats, on='user_id', how='left')
        features = features.merge(self.book_stats, on='book_id', how='left')
        
        # Простые но эффективные engineered features
        features['user_book_match'] = features['user_mean'] * features['book_mean'] / 10
        features['book_popularity'] = features['book_mean'] * np.log1p(features['book_count'])
        features['book_age'] = 2023 - features['publication_year']
        
        # Заполняем пропуски
        features['user_mean'] = features['user_mean'].fillna(features['user_mean'].median())
        features['book_mean'] = features['book_mean'].fillna(features['book_mean'].median())
        features['user_count'] = features['user_count'].fillna(1)
        features['book_count'] = features['book_count'].fillna(1)
        features['book_age'] = features['book_age'].fillna(20)
        features['age'] = features['age'].fillna(features['age'].median())
        features['gender'] = features['gender'].fillna(1)
        
        # Выбираем финальные признаки
        final_features = [
            'user_mean', 'book_mean', 'user_count', 'book_count',
            'user_book_match', 'book_popularity', 'book_age',
            'avg_rating', 'age', 'gender'
        ]
        
        print(f"✅ Используется {len(final_features)} признаков")
        
        return features[final_features]
    
    def train_model(self, X, y):
        """Обучение модели"""
        print("\n🎯 ОБУЧЕНИЕ МОДЕЛИ")
        
        # Простое разделение
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Простая но эффективная модель
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Валидация
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mae = mean_absolute_error(y_val, y_pred)
        
        print(f"📊 ВАЛИДАЦИЯ:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        
        # Финальное обучение
        print("🔥 Финальное обучение...")
        model.fit(X, y)
        self.model = model
        
        return model
    
    def smart_postprocess(self, predictions):
        """Умная пост-обработка на основе анализа данных"""
        print("\n🎨 УМНАЯ ПОСТ-ОБРАБОТКА")
        
        # Анализ: 72% оценок >= 8, поэтому сдвигаем ВСЕ предсказания вверх
        result = predictions.copy()
        
        # АГРЕССИВНЫЙ СДВИГ К ВЫСОКИМ ОЦЕНКАМ
        result = np.where(result < 9.0, result + 1.0, result)
        result = np.where(result < 8.0, result + 1.5, result)
        result = np.where(result < 7.0, result + 2.0, result)
        result = np.where(result < 6.0, result + 2.5, result)
        
        # Округляем до 0.5
        result = np.round(result * 2) / 2
        
        # Жестко ограничиваем диапазон
        result = np.clip(result, 1.0, 10.0)
        
        print(f"   Было: {predictions.mean():.2f} -> Стало: {result.mean():.2f}")
        
        return result
    
    def create_submission(self):
        """Создание сабмита"""
        print("\n🏆 СОЗДАНИЕ САБМИТА")
        
        # Подготовка теста
        X_test = self.create_smart_features(self.test, is_train=False)
        
        # Предсказание
        predictions = self.model.predict(X_test)
        
        # Умная пост-обработка
        final_predictions = self.smart_postprocess(predictions)
        
        # Создание сабмита
        submission = self.test[['user_id', 'book_id']].copy()
        submission['rating_predict'] = final_predictions
        
        # Сохранение
        submission.to_csv('submission_fixed.csv', index=False)
        print("✅ submission_fixed.csv создан")
        
        # Анализ
        stats = submission['rating_predict'].describe()
        print(f"\n📊 СТАТИСТИКА:")
        print(f"   Среднее: {stats['mean']:.2f}")
        print(f"   Медиана: {stats['50%']:.2f}")
        
        # Распределение
        print(f"\n🎯 РАСПРЕДЕЛЕНИЕ:")
        for rating in [8, 9, 10]:
            count = (submission['rating_predict'] == rating).sum()
            pct = count / len(submission) * 100
            print(f"   {rating}: {count:4d} ({pct:5.1f}%)")
        
        return submission
    
    def run_working_pipeline(self):
        """Запуск рабочего пайплайна"""
        print("=" * 50)
        print("🚀 РАБОЧИЙ ПАЙПЛАЙН")
        print("=" * 50)
        
        try:
            # 1. Загрузка
            self.load_data()
            
            # 2. Подготовка данных (БЕЗ ошибок индексов!)
            train_read = self.train[self.train['has_read'] == 1]
            X = self.create_smart_features(train_read, is_train=True)
            y = train_read['rating'].values  # Используем .values чтобы избежать проблем с индексами
            
            print(f"💪 Данные: X{X.shape}, y{y.shape}")
            
            # 3. Обучение
            self.train_model(X, y)
            
            # 4. Создание сабмита
            submission = self.create_submission()
            
            print("\n" + "=" * 50)
            print("🎉 ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
            print("=" * 50)
            
            return submission
            
        except Exception as e:
            print(f"❌ ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            
            # Аварийный сабмит
            submission = self.test[['user_id', 'book_id']].copy()
            submission['rating_predict'] = 9.0
            submission.to_csv('submission_simple.csv', index=False)
            return submission

# ЗАПУСК
if __name__ == "__main__":
    worker = WorkingRatingPredictor()
    submission = worker.run_working_pipeline()
    
    print("\n🔍 ПРИМЕР ПРЕДСКАЗАНИЙ:")
    print(submission.head(10))