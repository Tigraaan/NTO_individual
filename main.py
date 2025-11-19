import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SmartBookRatingPredictor:
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.label_encoders = {}
        
    def load_data(self):
        """Загрузка всех необходимых данных"""
        print("Загрузка данных...")
        
        self.train = pd.read_csv('train.csv', parse_dates=['timestamp'])
        self.test = pd.read_csv('test.csv')
        self.books = pd.read_csv('books.csv')
        self.users = pd.read_csv('users.csv')
        self.book_genres = pd.read_csv('book_genres.csv')
        
        print(f"Train: {len(self.train)} записей")
        print(f"Test: {len(self.test)} записей")
        print(f"Books: {len(self.books)} книг")
        print(f"Users: {len(self.users)} пользователей")
        
    def analyze_target_distribution(self):
        """Анализ распределения целевой переменной"""
        print("\n=== АНАЛИЗ РАСПРЕДЕЛЕНИЯ ОЦЕНОК ===")
        
        ratings = self.train[self.train['has_read'] == 1]['rating']
        
        print("Распределение оценок:")
        for i in range(1, 11):
            count = (ratings == i).sum()
            percentage = (ratings == i).mean() * 100
            print(f"Оценка {i}: {count:5d} записей ({percentage:5.1f}%)")
        
        print(f"\nОбщая статистика:")
        print(f"Среднее: {ratings.mean():.2f}")
        print(f"Медиана: {ratings.median():.2f}")
        print(f"Стандартное отклонение: {ratings.std():.2f}")
        
        # Особое внимание на высокие оценки
        high_ratings = (ratings >= 9).sum()
        print(f"\nОценки 9-10: {high_ratings} записей ({(ratings >= 9).mean()*100:.1f}%)")
        
        return ratings

    def create_smart_features(self, df, is_train=True):
        """Создание умных признаков, основанных на анализе данных"""
        features = df.copy()
        
        # Базовые объединения
        features = features.merge(self.books, on='book_id', how='left')
        features = features.merge(self.users, on='user_id', how='left')
        
        # === КРИТИЧЕСКИ ВАЖНЫЕ ПРИЗНАКИ ===
        
        # 1. Средний рейтинг книги (самый важный признак)
        features['book_avg_rating'] = features['avg_rating']
        
        # 2. Возраст книги
        features['book_age'] = 2023 - features['publication_year']
        features['book_age'] = features['book_age'].fillna(0).clip(0, 100)
        
        # 3. Количество жанров у книги
        book_genre_counts = self.book_genres.groupby('book_id').size().reset_index(name='genre_count')
        features = features.merge(book_genre_counts, on='book_id', how='left')
        features['genre_count'] = features['genre_count'].fillna(0)
        
        # 4. Статистики пользователей (только по ПРОЧИТАННЫМ книгам)
        if is_train:
            user_read_stats = self.train[self.train['has_read'] == 1].groupby('user_id').agg({
                'rating': ['mean', 'std', 'count'],
                'book_id': 'nunique'
            }).round(3)
            user_read_stats.columns = ['user_rating_mean', 'user_rating_std', 'user_books_rated', 'user_unique_books']
            self.user_read_stats = user_read_stats.reset_index()
            
            # Статистики по списку "хочу прочитать"
            user_wishlist_stats = self.train[self.train['has_read'] == 0].groupby('user_id').agg({
                'book_id': 'count'
            }).reset_index().rename(columns={'book_id': 'user_wishlist_count'})
            self.user_wishlist_stats = user_wishlist_stats
        
        # Добавляем пользовательские статистики
        features = features.merge(self.user_read_stats, on='user_id', how='left')
        features = features.merge(self.user_wishlist_stats, on='user_id', how='left')
        
        # 5. Популярность книги (сколько раз её читали)
        if is_train:
            book_popularity = self.train[self.train['has_read'] == 1].groupby('book_id').agg({
                'user_id': 'count'
            }).reset_index().rename(columns={'user_id': 'book_read_count'})
            self.book_popularity = book_popularity
        
        features = features.merge(self.book_popularity, on='book_id', how='left')
        
        # 6. Авторские статистики
        if is_train:
            author_stats = self.train[self.train['has_read'] == 1].merge(
                self.books[['book_id', 'author_id']], on='book_id'
            ).groupby('author_id').agg({
                'rating': ['mean', 'count']
            }).round(3)
            author_stats.columns = ['author_avg_rating', 'author_books_rated']
            self.author_stats = author_stats.reset_index()
        
        features = features.merge(self.author_stats, on='author_id', how='left')
        
        # 7. Временные признаки (только для train)
        if is_train and 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features['year'] = features['timestamp'].dt.year
            features['month'] = features['timestamp'].dt.month
        
        # === ЗАПОЛНЕНИЕ ПРОПУСКОВ ===
        
        # Для пользователей без истории - средние значения по всем пользователям
        features['user_rating_mean'] = features['user_rating_mean'].fillna(features['user_rating_mean'].mean())
        features['user_rating_std'] = features['user_rating_std'].fillna(0)
        features['user_books_rated'] = features['user_books_rated'].fillna(0)
        features['user_unique_books'] = features['user_unique_books'].fillna(0)
        features['user_wishlist_count'] = features['user_wishlist_count'].fillna(0)
        
        # Для книг без статистики
        features['book_read_count'] = features['book_read_count'].fillna(1)
        features['author_avg_rating'] = features['author_avg_rating'].fillna(features['book_avg_rating'].mean())
        features['author_books_rated'] = features['author_books_rated'].fillna(1)
        
        # Демографические признаки
        features['age'] = features['age'].fillna(features['age'].median())
        features['gender'] = features['gender'].fillna(1)  # предположим мужской
        
        # === ФИНАЛЬНЫЕ ПРИЗНАКИ ===
        
        final_features = [
            # Книжные признаки
            'book_avg_rating', 'book_age', 'genre_count', 'book_read_count',
            
            # Пользовательские признаки
            'user_rating_mean', 'user_rating_std', 'user_books_rated', 
            'user_unique_books', 'user_wishlist_count', 'age', 'gender',
            
            # Авторские признаки
            'author_avg_rating', 'author_books_rated',
            
            # Взаимодействия
            'publication_year', 'language', 'publisher'
        ]
        
        # Добавляем временные признаки для train
        if is_train:
            final_features.extend(['year', 'month'])
        
        # Оставляем только существующие колонки
        available_features = [f for f in final_features if f in features.columns]
        
        if is_train:
            self.feature_columns = available_features
        
        print(f"Используется {len(available_features)} признаков")
        
        return features[available_features]

    def prepare_data(self):
        """Подготовка данных для обучения"""
        print("\n=== ПОДГОТОВКА ДАННЫХ ===")
        
        # Анализ целевой переменной
        ratings = self.analyze_target_distribution()
        
        # Подготовка признаков для обучения
        X = self.create_smart_features(self.train[self.train['has_read'] == 1], is_train=True)
        y = self.train[self.train['has_read'] == 1]['rating']
        
        print(f"Данные для обучения: X.shape={X.shape}, y.shape={y.shape}")
        
        return X, y

    def train_optimized_model(self, X, y):
        """Обучение оптимизированной модели"""
        print("\n=== ОБУЧЕНИЕ МОДЕЛИ ===")
        
        # Разделение на train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Train: {X_train.shape}, Validation: {X_val.shape}")
        
        # Оптимизированная модель Random Forest
        model = RandomForestRegressor(
            n_estimators=200,           # Увеличили количество деревьев
            max_depth=25,               # Увеличили глубину
            min_samples_split=5,        # Уменьшили для большей гибкости
            min_samples_leaf=2,         # Уменьшили для лучшего обучения
            max_features='sqrt',        # Оптимально для Random Forest
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        print("Обучение модели...")
        model.fit(X_train, y_train)
        
        # Предсказание на валидации
        y_pred = model.predict(X_val)
        
        # Метрики
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mae = mean_absolute_error(y_val, y_pred)
        
        print(f"\nРезультаты на валидации:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # Анализ ошибок по диапазонам оценок
        self.analyze_prediction_errors(y_val, y_pred)
        
        # Важность признаков
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nТоп-10 самых важных признаков:")
        print(feature_importance.head(10))
        
        # Обучение на всех данных
        print("\nОбучение финальной модели на всех данных...")
        model.fit(X, y)
        self.models['rf'] = model
        
        return model

    def analyze_prediction_errors(self, y_true, y_pred):
        """Анализ ошибок предсказания"""
        errors = np.abs(y_true - y_pred)
        
        print("\nАнализ ошибок по диапазонам оценок:")
        for rating_range in [(1, 3), (4, 6), (7, 8), (9, 10)]:
            mask = (y_true >= rating_range[0]) & (y_true <= rating_range[1])
            if mask.any():
                range_mae = errors[mask].mean()
                count = mask.sum()
                print(f"Оценки {rating_range[0]}-{rating_range[1]}: MAE = {range_mae:.3f} ({count} samples)")

    def predict(self, test_df):
        """Предсказание для тестовых данных"""
        print("\n=== ПРЕДСКАЗАНИЕ ===")
        
        # Подготовка признаков для теста
        X_test = self.create_smart_features(test_df, is_train=False)
        
        # Предсказание
        predictions = self.models['rf'].predict(X_test)
        
        # Пост-обработка предсказаний на основе анализа данных
        # Учитываем, что большинство оценок высокие
        predictions = self.post_process_predictions(predictions)
        
        return predictions

    def post_process_predictions(self, predictions):
        """Пост-обработка предсказаний на основе анализа данных"""
        # Ограничиваем диапазон
        predictions = np.clip(predictions, 1, 10)
        
        # Сдвигаем предсказания в сторону высоких оценок (на основе анализа данных)
        # Большинство оценок в данных высокие, поэтому сдвигаем предсказания вверх
        predictions = np.where(predictions < 7, predictions + 0.5, predictions)
        predictions = np.where(predictions > 9.5, 9.5, predictions)
        
        # Округляем до 0.5 для более естественного вида
        predictions = np.round(predictions * 2) / 2
        
        return predictions

    def create_submission(self, test_df, predictions, filename='submission_smart.csv'):
        """Создание файла для отправки"""
        print("\n=== СОЗДАНИЕ САБМИТА ===")
        
        submission = test_df[['user_id', 'book_id']].copy()
        submission['rating_predict'] = predictions
        
        # Сохраняем файл
        submission.to_csv(filename, index=False)
        print(f"Файл {filename} создан")
        
        # Детальная статистика предсказаний
        print(f"\nСтатистика предсказаний:")
        print(f"Min: {submission['rating_predict'].min():.3f}")
        print(f"Max: {submission['rating_predict'].max():.3f}")
        print(f"Mean: {submission['rating_predict'].mean():.3f}")
        print(f"Std: {submission['rating_predict'].std():.3f}")
        
        # Распределение предсказаний
        print(f"\nРаспределение предсказаний:")
        for i in range(1, 11):
            count = (submission['rating_predict'] == i).sum()
            percentage = (submission['rating_predict'] == i).mean() * 100
            if count > 0:
                print(f"Оценка {i}: {count:4d} записей ({percentage:5.1f}%)")
        
        return submission

    def run_smart_pipeline(self):
        """Запуск умного пайплайна"""
        print("=== ЗАПУСК УМНОГО ПАЙПЛАЙНА ===")
        
        try:
            # 1. Загрузка данных
            self.load_data()
            
            # 2. Подготовка данных
            X, y = self.prepare_data()
            
            # 3. Обучение модели
            self.train_optimized_model(X, y)
            
            # 4. Предсказание
            predictions = self.predict(self.test)
            
            # 5. Создание сабмита
            submission = self.create_submission(self.test, predictions)
            
            print("\n🎉 ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
            
            return submission
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            
            # Резервный сабмит
            submission = self.test[['user_id', 'book_id']].copy()
            submission['rating_predict'] = 8.0  # Средняя высокая оценка
            submission.to_csv('submission_backup.csv', index=False)
            print("Создан резервный submission файл")
            return submission

# Запуск программы
if __name__ == "__main__":
    predictor = SmartBookRatingPredictor()
    submission = predictor.run_smart_pipeline()
    
    print("\n" + "="*50)
    print("Sample предсказаний:")
    print(submission.head(10))
    print("="*50)