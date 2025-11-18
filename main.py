# ultimate_professional_solution.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Для красивого прогресс-бара
from tqdm import tqdm
import time

class UltimateProfessionalPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    def auto_detect_columns(self, df, df_type):
        """Автоматическое определение колонок с умным поиском"""
        column_map = {}
        
        # Все возможные варианты названий для каждого типа колонок
        user_keywords = ['user', 'id', 'client', 'person', 'customer']
        book_keywords = ['book', 'item', 'product', 'movie', 'article'] 
        rating_keywords = ['rating', 'score', 'target', 'label', 'eval']
        read_keywords = ['read', 'has', 'interaction', 'action']
        
        # Поиск user_id
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in user_keywords):
                if 'book' not in col_lower and 'item' not in col_lower:
                    column_map['user_id'] = col
                    break
        else:
            column_map['user_id'] = df.columns[0]  # Первая колонка по умолчанию
        
        # Поиск book_id
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in book_keywords):
                column_map['book_id'] = col
                break
        else:
            # Вторая колонка или первая если только одна колонка
            column_map['book_id'] = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # Для train данных ищем rating и has_read
        if df_type == 'train':
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in rating_keywords):
                    column_map['rating'] = col
                    break
            
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in read_keywords):
                    column_map['has_read'] = col
                    break
        
        return column_map
    
    def load_and_prepare_data(self):
        """Загрузка и подготовка данных с детальным логированием"""
        print("📂 ЗАГРУЗКА И АНАЛИЗ ДАННЫХ...")
        
        # Загрузка с прогресс-баром
        files = [
            ('train.csv', 'Обучающие данные'),
            ('test.csv', 'Тестовые данные'),
        ]
        
        data = {}
        
        for filename, description in tqdm(files, desc="Загрузка файлов"):
            try:
                # Пробуем разные разделители
                for sep in [';', ',', '\t']:
                    try:
                        df = pd.read_csv(filename, sep=sep)
                        if len(df.columns) > 1:  # Убедимся что есть несколько колонок
                            data[filename.replace('.csv', '')] = df
                            print(f"   ✅ {description} загружены (разделитель: '{sep}')")
                            break
                    except:
                        continue
                else:
                    print(f"   ❌ Не удалось загрузить {filename}")
                    data[filename.replace('.csv', '')] = None
            except Exception as e:
                print(f"   ❌ Ошибка загрузки {filename}: {e}")
                data[filename.replace('.csv', '')] = None
        
        if data['train'] is None:
            raise Exception("Не удалось загрузить train.csv")
        
        # Автоматическое определение колонок
        print("\n🎯 АНАЛИЗ СТРУКТУРЫ ДАННЫХ...")
        train_columns = self.auto_detect_columns(data['train'], 'train')
        print(f"   Train колонки: {train_columns}")
        
        if data['test'] is not None:
            test_columns = self.auto_detect_columns(data['test'], 'test')
            print(f"   Test колонки: {test_columns}")
        
        # Переименование колонок
        data['train'] = data['train'].rename(columns=train_columns)
        if data['test'] is not None:
            data['test'] = data['test'].rename(columns=test_columns)
        
        return data['train'], data['test']
    
    def create_features_with_progress(self, df, is_train=True):
        """Создание признаков с визуализацией процесса"""
        print("\n🔧 СОЗДАНИЕ ПРИЗНАКОВ...")
        
        steps = [
            "Подготовка данных",
            "Статистики пользователей", 
            "Статистики книг",
            "Инженерные признаки",
            "Взаимодействия",
            "Финальная обработка"
        ]
        
        pbar = tqdm(total=len(steps), desc="Прогресс создания признаков")
        
        try:
            # Шаг 1: Подготовка данных
            pbar.set_description("📊 Подготовка данных")
            if is_train and 'has_read' in df.columns and 'rating' in df.columns:
                df = df[df['has_read'] == 1].copy()
                print(f"   📖 Используем {len(df)} прочитанных книг")
            time.sleep(0.3)
            pbar.update(1)
            
            # Шаг 2: Статистики пользователей
            pbar.set_description("👤 Статистики пользователей")
            if is_train and 'rating' in df.columns:
                self.user_stats = df.groupby('user_id').agg({
                    'rating': ['mean', 'count', 'std', 'min', 'max', 'median']
                }).reset_index()
                self.user_stats.columns = ['user_id', 'user_mean', 'user_count', 'user_std', 'user_min', 'user_max', 'user_median']
                self.global_mean = df['rating'].mean()
                self.global_std = df['rating'].std()
                print(f"   📈 Пользователей: {len(self.user_stats)}")
            time.sleep(0.3)
            pbar.update(1)
            
            # Шаг 3: Статистики книг
            pbar.set_description("📚 Статистики книг")
            if is_train and 'rating' in df.columns:
                self.book_stats = df.groupby('book_id').agg({
                    'rating': ['mean', 'count', 'std', 'min', 'max', 'median']
                }).reset_index()
                self.book_stats.columns = ['book_id', 'book_mean', 'book_count', 'book_std', 'book_min', 'book_max', 'book_median']
                print(f"   📊 Книг: {len(self.book_stats)}")
            time.sleep(0.3)
            pbar.update(1)
            
            # Шаг 4: Объединение и базовые признаки
            pbar.set_description("🔄 Объединение данных")
            df = df.merge(self.user_stats, on='user_id', how='left')
            df = df.merge(self.book_stats, on='book_id', how='left')
            
            # Заполнение пропусков
            stats_to_fill = {
                'user_mean': self.global_mean, 'user_count': 1, 'user_std': self.global_std,
                'user_min': 1.0, 'user_max': 10.0, 'user_median': self.global_mean,
                'book_mean': self.global_mean, 'book_count': 1, 'book_std': self.global_std,
                'book_min': 1.0, 'book_max': 10.0, 'book_median': self.global_mean
            }
            
            for col, fill_val in stats_to_fill.items():
                if col in df.columns:
                    df[col] = df[col].fillna(fill_val)
            time.sleep(0.3)
            pbar.update(1)
            
            # Шаг 5: Инженерные признаки
            pbar.set_description("⚙️ Инженерные признаки")
            # User features
            df['user_confidence'] = np.log1p(df['user_count']) / 4.0
            df['user_generosity'] = (df['user_mean'] - self.global_mean) / max(self.global_std, 0.1)
            df['user_consistency'] = 1 / (1 + df['user_std'].fillna(1))
            
            # Book features
            df['book_popularity'] = np.log1p(df['book_count']) / 4.0
            df['book_controversial'] = (df['book_std'] > 2.0).astype(int)
            df['book_consistency'] = 1 / (1 + df['book_std'].fillna(1))
            time.sleep(0.3)
            pbar.update(1)
            
            # Шаг 6: Взаимодействия и финальная обработка
            pbar.set_description("🎯 Финальная обработка")
            # Interaction features
            df['mean_interaction'] = df['user_mean'] * df['book_mean'] / 10.0
            df['confidence_interaction'] = df['user_confidence'] * df['book_popularity']
            df['prediction_baseline'] = 0.6 * df['user_mean'] + 0.4 * df['book_mean']
            
            # Финальный набор признаков
            feature_columns = [
                'user_mean', 'user_count', 'user_std', 'user_min', 'user_max', 'user_median',
                'book_mean', 'book_count', 'book_std', 'book_min', 'book_max', 'book_median',
                'user_confidence', 'user_generosity', 'user_consistency',
                'book_popularity', 'book_controversial', 'book_consistency',
                'mean_interaction', 'confidence_interaction', 'prediction_baseline'
            ]
            
            available_features = [f for f in feature_columns if f in df.columns]
            df[available_features] = df[available_features].fillna(0)
            
            if is_train:
                self.feature_columns = available_features
            
            pbar.update(1)
            pbar.close()
            
            print(f"   ✅ Создано {len(self.feature_columns)} признаков")
            return df[available_features]
            
        except Exception as e:
            pbar.close()
            raise e
    
    def train_with_detailed_progress(self, X, y):
        """Обучение с детальным выводом прогресса"""
        print("\n🚀 НАЧАЛО ОБУЧЕНИЯ МОДЕЛЕЙ")
        print("=" * 50)
        
        # Разделение данных
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"📊 РАЗМЕРНОСТИ ДАННЫХ:")
        print(f"   Обучающая выборка: {X_train.shape}")
        print(f"   Валидационная выборка: {X_val.shape}")
        
        # Масштабирование
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_val_scaled = self.scalers['standard'].transform(X_val)
        
        models_performance = []
        
        # 1. Gradient Boosting с прогрессом
        print("\n🔥 ОБУЧЕНИЕ GRADIENT BOOSTING")
        print("   " + "─" * 40)
        
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42,
            verbose=1
        )
        
        print("   🎯 Начало обучения...")
        gb_model.fit(X_train_scaled, y_train)
        self.models['gb'] = gb_model
        
        # Оценка качества
        train_pred_gb = gb_model.predict(X_train_scaled)
        val_pred_gb = gb_model.predict(X_val_scaled)
        
        train_rmse_gb = np.sqrt(mean_squared_error(y_train, train_pred_gb))
        val_rmse_gb = np.sqrt(mean_squared_error(y_val, val_pred_gb))
        train_mae_gb = mean_absolute_error(y_train, train_pred_gb)
        val_mae_gb = mean_absolute_error(y_val, val_pred_gb)
        
        models_performance.append(('Gradient Boosting', val_rmse_gb, val_mae_gb))
        
        print(f"   📈 Результаты Gradient Boosting:")
        print(f"     Train RMSE: {train_rmse_gb:.4f} | Val RMSE: {val_rmse_gb:.4f}")
        print(f"     Train MAE:  {train_mae_gb:.4f} | Val MAE:  {val_mae_gb:.4f}")
        
        # 2. Random Forest с прогресс-баром
        print("\n🌳 ОБУЧЕНИЕ RANDOM FOREST")
        print("   " + "─" * 40)
        
        rf_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            min_samples_split=30,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print("   🎯 Построение деревьев...")
        rf_model.fit(X_train, y_train)
        self.models['rf'] = rf_model
        
        # Оценка качества
        val_pred_rf = rf_model.predict(X_val)
        val_rmse_rf = np.sqrt(mean_squared_error(y_val, val_pred_rf))
        val_mae_rf = mean_absolute_error(y_val, val_pred_rf)
        
        models_performance.append(('Random Forest', val_rmse_rf, val_mae_rf))
        
        print(f"   📈 Результаты Random Forest:")
        print(f"     Val RMSE: {val_rmse_rf:.4f} | Val MAE: {val_mae_rf:.4f}")
        
        # 3. Ridge Regression
        print("\n📐 ОБУЧЕНИЕ RIDGE REGRESSION")
        print("   " + "─" * 40)
        
        ridge_model = Ridge(alpha=1.0, random_state=42)
        ridge_model.fit(X_train_scaled, y_train)
        self.models['ridge'] = ridge_model
        
        # Оценка качества
        val_pred_ridge = ridge_model.predict(X_val_scaled)
        val_rmse_ridge = np.sqrt(mean_squared_error(y_val, val_pred_ridge))
        val_mae_ridge = mean_absolute_error(y_val, val_pred_ridge)
        
        models_performance.append(('Ridge Regression', val_rmse_ridge, val_mae_ridge))
        
        print(f"   📈 Результаты Ridge Regression:")
        print(f"     Val RMSE: {val_rmse_ridge:.4f} | Val MAE: {val_mae_ridge:.4f}")
        
        # Сравнение моделей
        print("\n🏆 ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
        print("   " + "=" * 50)
        print(f"   {'МОДЕЛЬ':<20} {'RMSE':<10} {'MAE':<10}")
        print("   " + "─" * 50)
        
        for name, rmse, mae in sorted(models_performance, key=lambda x: x[1]):
            print(f"   🎯 {name:<18} {rmse:<10.4f} {mae:<10.4f}")
        
        best_model = min(models_performance, key=lambda x: x[1])
        print(f"\n   💪 ЛУЧШАЯ МОДЕЛЬ: {best_model[0]}")
        print(f"   📊 Лучший RMSE: {best_model[1]:.4f}")
        
        return best_model[1]
    
    def predict_ensemble(self, X):
        """Предсказание ансамблем моделей"""
        if len(X) == 0:
            return np.array([self.global_mean] * len(X))
        
        X_scaled = self.scalers['standard'].transform(X)
        
        preds_gb = self.models['gb'].predict(X_scaled)
        preds_rf = self.models['rf'].predict(X)
        preds_ridge = self.models['ridge'].predict(X_scaled)
        
        # Взвешенное усреднение
        weights = {'gb': 0.5, 'rf': 0.3, 'ridge': 0.2}
        ensemble_pred = (
            weights['gb'] * preds_gb + 
            weights['rf'] * preds_rf + 
            weights['ridge'] * preds_ridge
        )
        
        return ensemble_pred
    
    def run_ultimate_solution(self):
        """Запуск ультимативного решения"""
        print("🎯 УЛЬТИМАТИВНОЕ ПРОФЕССИОНАЛЬНОЕ РЕШЕНИЕ")
        print("💡 С автоматическим определением данных и детальным обучением")
        print("=" * 70)
        
        try:
            # 1. Загрузка и подготовка данных
            train, test = self.load_and_prepare_data()
            
            # 2. Создание признаков
            X_train = self.create_features_with_progress(train, is_train=True)
            
            # Получаем целевую переменную
            if 'has_read' in train.columns and 'rating' in train.columns:
                y_train = train[train['has_read'] == 1]['rating']
            elif 'rating' in train.columns:
                y_train = train['rating']
            else:
                # Если нет рейтингов, создаем искусственные
                y_train = pd.Series([self.global_mean] * len(X_train))
                print("   ⚠️ Рейтинги не найдены, используем базовые значения")
            
            print(f"\n📊 ФИНАЛЬНЫЕ ДАННЫЕ ДЛЯ ОБУЧЕНИЯ:")
            print(f"   Признаки: {X_train.shape}")
            print(f"   Целевая переменная: {len(y_train)}")
            
            # 3. Обучение моделей
            best_rmse = self.train_with_detailed_progress(X_train, y_train)
            
            # 4. Предсказание на тесте
            if test is not None:
                print("\n🎯 ГЕНЕРАЦИЯ ПРЕДСКАЗАНИЙ...")
                X_test = self.create_features_with_progress(test, is_train=False)
                X_test = X_test.fillna(0)
                
                # Прогресс-бар для предсказаний
                predictions = []
                for i in tqdm(range(len(X_test)), desc="Создание предсказаний", unit="запись"):
                    pred = self.predict_ensemble(X_test.iloc[i:i+1])
                    predictions.append(pred[0])
                    time.sleep(0.001)  # Для плавного прогресса
                
                predictions = np.clip(predictions, 1.0, 10.0)
                
                # Создание сабмита
                submission = test[['user_id', 'book_id']].copy()
                submission['rating_predict'] = predictions
                
                submission.to_csv('ultimate_professional_submission.csv', index=False)
                
                print(f"\n💾 САБМИТ СОХРАНЕН: ultimate_professional_submission.csv")
                print(f"📊 Качество модели: RMSE = {best_rmse:.4f}")
                
                return submission
            else:
                raise Exception("Тестовые данные не найдены")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            return None

# ЗАПУСК
if __name__ == "__main__":
    predictor = UltimateProfessionalPredictor()
    submission = predictor.run_ultimate_solution()
    
    if submission is not None:
        print(f"\n🎉 УЛЬТИМАТИВНОЕ РЕШЕНИЕ УСПЕШНО СОЗДАНО!")
        print("📤 Отправляйте: ultimate_professional_submission.csv")
    else:
        print("\n❌ Не удалось создать решение")
    
    print("💪 УДАЧИ В СОРЕВНОВАНИИ!")