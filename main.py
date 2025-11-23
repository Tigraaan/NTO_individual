import pandas as pd
import numpy as np

class FinalBoss:
    def __init__(self):
        pass
    
    def load_data(self):
        """БЫСТРАЯ ЗАГРУЗКА"""
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        return train[train['has_read'] == 1], test
    
    def analyze_cold_start(self, train_read, test):
        """АНАЛИЗ ХОЛОДНОГО СТАРТА"""
        train_users = set(train_read['user_id'])
        train_books = set(train_read['book_id'])
        test_users = set(test['user_id'])
        test_books = set(test['book_id'])
        
        new_users = len(test_users - train_users)
        new_books = len(test_books - train_books)
        
        print(f"🔥 НОВЫЕ ПОЛЬЗОВАТЕЛИ: {new_users}/{len(test_users)} ({new_users/len(test_users)*100:.1f}%)")
        print(f"🔥 НОВЫЕ КНИГИ: {new_books}/{len(test_books)} ({new_books/len(test_books)*100:.1f}%)")
        
        return new_users > 0 or new_books > 0
    
    def create_smart_baseline(self, train_read, test):
        """УМНЫЙ БЕЙЗЛАЙН"""
        # ГЛОБАЛЬНОЕ СРЕДНЕЕ
        global_mean = train_read['rating'].mean()
        
        # СРЕДНИЕ ПОЛЬЗОВАТЕЛЕЙ И КНИГ
        user_means = train_read.groupby('user_id')['rating'].mean()
        book_means = train_read.groupby('book_id')['rating'].mean()
        
        predictions = []
        for _, row in test.iterrows():
            user_id, book_id = row['user_id'], row['book_id']
            
            user_known = user_id in user_means.index
            book_known = book_id in book_means.index
            
            if user_known and book_known:
                pred = (user_means[user_id] + book_means[book_id]) / 2
            elif user_known:
                pred = user_means[user_id]
            elif book_known:
                pred = book_means[book_id]
            else:
                pred = global_mean
                
            predictions.append(pred)
            
        return np.array(predictions)
    
    def optimize_predictions(self, predictions):
        """ФИНАЛЬНАЯ ОПТИМИЗАЦИЯ"""
        # ЦЕЛЕВОЕ РАСПРЕДЕЛЕНИЕ НА ОСНОВЕ ТРЕНИРОВОЧНЫХ ДАННЫХ
        target_mean = 8.7  # Чуть выше среднего
        target_std = 1.2   # Немного вариативности
        
        # НОРМАЛИЗУЕМ И ПЕРЕМАСШТАБИРУЕМ
        current_mean = np.mean(predictions)
        current_std = np.std(predictions)
        
        if current_std > 0:
            normalized = (predictions - current_mean) / current_std
            optimized = normalized * target_std + target_mean
        else:
            optimized = np.full_like(predictions, target_mean)
        
        # ОГРАНИЧИВАЕМ И ОКРУГЛЯЕМ
        optimized = np.clip(optimized, 1.0, 10.0)
        optimized = np.round(optimized * 2) / 2  # До 0.5
        
        return optimized
    
    def create_final_submission(self, test, predictions):
        """ФИНАЛЬНЫЙ САБМИТ"""
        submission = test[['user_id', 'book_id']].copy()
        submission['rating_predict'] = predictions
        submission.to_csv('submission_FINAL_BOSS.csv', index=False)
        
        # СТАТИСТИКА
        stats = submission['rating_predict'].describe()
        print(f"📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
        print(f"• Mean: {stats['mean']:.3f}")
        print(f"• Std: {stats['std']:.3f}")
        print(f"• Min: {stats['min']:.1f}, Max: {stats['max']:.1f}")
        
        return submission
    
    def execute(self):
        """ВЫПОЛНЕНИЕ"""
        print("=" * 50)
        print("💀 FINAL BOSS MODE ACTIVATED")
        print("=" * 50)
        
        train_read, test = self.load_data()
        
        # ПРОВЕРЯЕМ ХОЛОДНЫЙ СТАРТ
        has_cold_start = self.analyze_cold_start(train_read, test)
        if has_cold_start:
            print("⚠️  ВНИМАНИЕ: Есть новые пользователи/книги!")
        
        # СОЗДАЕМ ПРЕДСКАЗАНИЯ
        predictions = self.create_smart_baseline(train_read, test)
        
        # ОПТИМИЗИРУЕМ
        final_predictions = self.optimize_predictions(predictions)
        
        # СОЗДАЕМ САБМИТ
        submission = self.create_final_submission(test, final_predictions)
        
        print("=" * 50)
        print("🎯 MISSION COMPLETE")
        print("=" * 50)
        
        return submission

# ЗАПУСК
if __name__ == "__main__":
    boss = FinalBoss()
    submission = boss.execute()
    print(submission.head())