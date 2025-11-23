import pandas as pd
import numpy as np
from scipy import stats

class PrecisionPredictor:
    def __init__(self):
        pass
    
    def load_and_analyze(self):
        """Глубокий анализ данных"""
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        books = pd.read_csv('books.csv')
        users = pd.read_csv('users.csv')
        
        train_read = train[train['has_read'] == 1]
        
        # ТОЧНЫЙ АНАЛИЗ РАСПРЕДЕЛЕНИЯ
        ratings = train_read['rating']
        print("🎯 ТОЧНОЕ РАСПРЕДЕЛЕНИЕ ОЦЕНОК:")
        for i in np.arange(1, 10.5, 0.5):
            count = ((ratings >= i - 0.25) & (ratings < i + 0.25)).sum()
            pct = count / len(ratings) * 100
            if count > 0:
                print(f"  {i:3.1f}: {count:6d} ({pct:5.1f}%)")
        
        return train_read, test, books, users
    
    def calculate_precise_means(self, train_read):
        """Точные статистики с доверительными интервалами"""
        # USER STATS
        user_stats = train_read.groupby('user_id').agg({
            'rating': ['mean', 'count', 'std'],
            'book_id': 'nunique'
        })
        user_stats.columns = ['user_mean', 'user_count', 'user_std', 'user_uniq_books']
        
        # Регуляризация для пользователей с малым количеством оценок
        global_mean = train_read['rating'].mean()
        user_stats['user_mean_reg'] = (
            (user_stats['user_mean'] * user_stats['user_count'] + global_mean * 5) / 
            (user_stats['user_count'] + 5)
        )
        
        # BOOK STATS
        book_stats = train_read.groupby('book_id').agg({
            'rating': ['mean', 'count', 'std'],
            'user_id': 'nunique'
        })
        book_stats.columns = ['book_mean', 'book_count', 'book_std', 'book_uniq_users']
        
        # Регуляризация для книг с малым количеством оценок
        book_stats['book_mean_reg'] = (
            (book_stats['book_mean'] * book_stats['book_count'] + global_mean * 3) / 
            (book_stats['book_count'] + 3)
        )
        
        return user_stats.reset_index(), book_stats.reset_index(), global_mean
    
    def bayesian_prediction(self, test, user_stats, book_stats, global_mean):
        """Байесовское предсказание с регуляризацией"""
        predictions = []
        confidence_scores = []
        
        for _, row in test.iterrows():
            user_id, book_id = row['user_id'], row['book_id']
            
            # Ищем данные
            user_data = user_stats[user_stats['user_id'] == user_id]
            book_data = book_stats[book_stats['book_id'] == book_id]
            
            # БАЙЕСОВСКАЯ КОМБИНАЦИЯ
            if len(user_data) > 0 and len(book_data) > 0:
                user_mean = user_data['user_mean_reg'].iloc[0]
                user_count = user_data['user_count'].iloc[0]
                book_mean = book_data['book_mean_reg'].iloc[0]
                book_count = book_data['book_count'].iloc[0]
                
                # ВЕСА НА ОСНОВЕ ДОСТОВЕРНОСТИ ДАННЫХ
                user_weight = np.log1p(user_count) / 10  # 0-1 вес
                book_weight = np.log1p(book_count) / 10  # 0-1 вес
                
                # НОРМАЛИЗАЦИЯ ВЕСОВ
                total_weight = user_weight + book_weight
                if total_weight > 0:
                    user_weight /= total_weight
                    book_weight /= total_weight
                else:
                    user_weight, book_weight = 0.4, 0.6
                
                prediction = user_mean * user_weight + book_mean * book_weight
                confidence = min(1.0, (user_weight + book_weight) / 2)
                
            elif len(user_data) > 0:
                user_mean = user_data['user_mean_reg'].iloc[0]
                user_count = user_data['user_count'].iloc[0]
                prediction = user_mean * 0.8 + global_mean * 0.2
                confidence = min(1.0, np.log1p(user_count) / 10)
                
            elif len(book_data) > 0:
                book_mean = book_data['book_mean_reg'].iloc[0]
                book_count = book_data['book_count'].iloc[0]
                prediction = book_mean * 0.8 + global_mean * 0.2
                confidence = min(1.0, np.log1p(book_count) / 10)
                
            else:
                prediction = global_mean
                confidence = 0.0
                
            predictions.append(prediction)
            confidence_scores.append(confidence)
            
        return np.array(predictions), np.array(confidence_scores)
    
    def optimize_distribution(self, predictions, confidence_scores):
        """Оптимизация распределения под метрики"""
        # АНАЛИЗ ТЕКУЩЕГО РАСПРЕДЕЛЕНИЯ
        current_stats = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'q25': np.percentile(predictions, 25),
            'q50': np.percentile(predictions, 50),
            'q75': np.percentile(predictions, 75)
        }
        
        print(f"📊 ТЕКУЩЕЕ РАСПРЕДЕЛЕНИЕ:")
        print(f"  Mean: {current_stats['mean']:.3f}")
        print(f"  Std:  {current_stats['std']:.3f}")
        print(f"  25%:  {current_stats['q25']:.2f}, 50%: {current_stats['q50']:.2f}, 75%: {current_stats['q75']:.2f}")
        
        # ЦЕЛЕВОЕ РАСПРЕДЕЛЕНИЕ (основано на реальных данных)
        target_mean = 8.55
        target_std = 1.25
        
        # ПРЕОБРАЗОВАНИЕ С СОХРАНЕНИЕМ ОТНОСИТЕЛЬНЫХ РАЗЛИЧИЙ
        if current_stats['std'] > 0:
            z_scores = (predictions - current_stats['mean']) / current_stats['std']
            optimized = z_scores * target_std + target_mean
        else:
            optimized = np.full_like(predictions, target_mean)
        
        # УЧЕТ ДОВЕРИЯ: для низкодостоверных предсказаний ближе к среднему
        confidence_adjustment = (optimized - target_mean) * (1 - confidence_scores)
        optimized = target_mean + confidence_adjustment
        
        # ПОСТ-ОБРАБОТКА
        optimized = np.clip(optimized, 1.0, 10.0)
        
        # ТОЧНОЕ ОКРУГЛЕНИЕ ДО 0.1 (больше точности)
        optimized = np.round(optimized * 10) / 10
        
        print(f"📊 ОПТИМИЗИРОВАННОЕ РАСПРЕДЕЛЕНИЕ:")
        print(f"  Mean: {np.mean(optimized):.3f}")
        print(f"  Std:  {np.std(optimized):.3f}")
        
        return optimized
    
    def create_precision_submission(self, test, predictions):
        """Создание точного сабмита"""
        submission = test[['user_id', 'book_id']].copy()
        submission['rating_predict'] = predictions
        submission.to_csv('submission_PRECISION.csv', index=False)
        
        # ДЕТАЛЬНЫЙ АНАЛИЗ
        stats = submission['rating_predict'].describe()
        print(f"\n🎯 ФИНАЛЬНАЯ СТАТИСТИКА:")
        print(f"• Count:  {stats['count']:.0f}")
        print(f"• Mean:   {stats['mean']:.3f}")
        print(f"• Std:    {stats['std']:.3f}")
        print(f"• Min:    {stats['min']:.2f}")
        print(f"• 25%:    {stats['25%']:.2f}")
        print(f"• 50%:    {stats['50%']:.2f}")
        print(f"• 75%:    {stats['75%']:.2f}")
        print(f"• Max:    {stats['max']:.2f}")
        
        # РАСПРЕДЕЛЕНИЕ С ШАГОМ 0.5
        print(f"\n📈 РАСПРЕДЕЛЕНИЕ:")
        for i in np.arange(1, 10.5, 0.5):
            count = ((submission['rating_predict'] >= i - 0.25) & 
                    (submission['rating_predict'] < i + 0.25)).sum()
            pct = count / len(submission) * 100
            if count > 0:
                print(f"  {i:3.1f}: {count:4d} ({pct:5.1f}%)")
        
        return submission
    
    def run_precision_model(self):
        """Запуск точной модели"""
        print("=" * 60)
        print("🎯 PRECISION PREDICTOR v3.0")
        print("=" * 60)
        
        train_read, test, books, users = self.load_and_analyze()
        
        # ТОЧНЫЕ СТАТИСТИКИ
        user_stats, book_stats, global_mean = self.calculate_precise_means(train_read)
        
        # БАЙЕСОВСКОЕ ПРЕДСКАЗАНИЕ
        predictions, confidence_scores = self.bayesian_prediction(test, user_stats, book_stats, global_mean)
        
        # ОПТИМИЗАЦИЯ РАСПРЕДЕЛЕНИЯ
        final_predictions = self.optimize_distribution(predictions, confidence_scores)
        
        # САБМИТ
        submission = self.create_precision_submission(test, final_predictions)
        
        print("=" * 60)
        print("💪 PRECISION MISSION COMPLETE!")
        print("=" * 60)
        
        return submission

# ЗАПУСК ТОЧНОЙ МОДЕЛИ
if __name__ == "__main__":
    precision = PrecisionPredictor()
    submission = precision.run_precision_model()
    print("\n🔍 ПРИМЕР ПРЕДСКАЗАНИЙ:")
    print(submission.head(10))