import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("🚀 ЗАПУСК УПРОЩЕННОГО РЕШЕНИЯ...")

# Загрузка данных с правильными параметрами
print("📊 Загрузка данных...")
try:
    # Пробуем разные варианты разделителей
    train = pd.read_csv('train.csv', sep=',', quotechar='"')
    print("✅ train.csv загружен с разделителем ','")
except:
    try:
        train = pd.read_csv('train.csv', sep=';', quotechar='"')
        print("✅ train.csv загружен с разделителем ';'")
    except:
        try:
            train = pd.read_csv('train.csv', delimiter='\t')
            print("✅ train.csv загружен с разделителем '\\t'")
        except Exception as e:
            print(f"❌ Не удалось загрузить train.csv: {e}")
            exit()

try:
    test = pd.read_csv('test.csv', sep=',', quotechar='"')
    print("✅ test.csv загружен с разделителем ','")
except:
    try:
        test = pd.read_csv('test.csv', sep=';', quotechar='"')
        print("✅ test.csv загружен с разделителем ';'")
    except:
        try:
            test = pd.read_csv('test.csv', delimiter='\t')
            print("✅ test.csv загружен с разделителем '\\t'")
        except Exception as e:
            print(f"❌ Не удалось загрузить test.csv: {e}")
            exit()

try:
    books = pd.read_csv('books.csv', sep=',', quotechar='"')
    print("✅ books.csv загружен с разделителем ','")
except:
    try:
        books = pd.read_csv('books.csv', sep=';', quotechar='"')
        print("✅ books.csv загружен с разделителем ';'")
    except:
        try:
            books = pd.read_csv('books.csv', delimiter='\t')
            print("✅ books.csv загружен с разделителем '\\t'")
        except Exception as e:
            print(f"❌ Не удалось загрузить books.csv: {e}")
            exit()

try:
    users = pd.read_csv('users.csv', sep=',', quotechar='"')
    print("✅ users.csv загружен с разделителем ','")
except:
    try:
        users = pd.read_csv('users.csv', sep=';', quotechar='"')
        print("✅ users.csv загружен с разделителем ';'")
    except:
        try:
            users = pd.read_csv('users.csv', delimiter='\t')
            print("✅ users.csv загружен с разделителем '\\t'")
        except Exception as e:
            print(f"❌ Не удалось загрузить users.csv: {e}")
            exit()

# Показать структуру данных
print("\n📋 СТРУКТУРА ДАННЫХ:")
print("train shape:", train.shape)
print("train columns:", train.columns.tolist())
print("\nПервые 3 строки train:")
print(train.head(3))

print("\ntest shape:", test.shape)
print("test columns:", test.columns.tolist())
print("\nbooks shape:", books.shape)
print("books columns:", books.columns.tolist())
print("\nusers shape:", users.shape)
print("users columns:", users.columns.tolist())

# Проверим наличие временной метки и пропустим если проблемы
timestamp_col = None
for col in train.columns:
    if 'time' in col.lower() or 'date' in col.lower():
        timestamp_col = col
        break

if timestamp_col:
    print(f"\n⏰ Временная метка найдена: '{timestamp_col}'")
    try:
        # Пробуем преобразовать только первые несколько строк для проверки
        sample_times = train[timestamp_col].head(10)
        print("Примеры временных меток:", sample_times.tolist())
        
        # Если есть проблемы, пропускаем временные признаки
        train[timestamp_col] = pd.to_datetime(train[timestamp_col], errors='coerce')
        print("✅ Временные метки преобразованы")
    except:
        print("⚠️ Проблемы с временными метками, работаем без них")
        timestamp_col = None
else:
    print("⚠️ Временная метка не найдена")

print("🔧 СОЗДАНИЕ ПРИЗНАКОВ...")

# БАЗОВЫЕ ПРИЗНАКИ (упрощенные)
def create_base_features(df, train_df=None):
    if train_df is None:
        return df
    
    # Агрегации по пользователям
    user_stats = train_df.groupby('user_id').agg({
        'rating': ['mean', 'count']
    }).round(3)
    user_stats.columns = ['user_mean_rating', 'user_rating_count']
    user_stats = user_stats.reset_index()
    
    # Агрегации по книгам  
    book_stats = train_df.groupby('book_id').agg({
        'rating': ['mean', 'count']
    }).round(3)
    book_stats.columns = ['book_mean_rating', 'book_rating_count']
    book_stats = book_stats.reset_index()
    
    # Объединение
    df = df.merge(user_stats, on='user_id', how='left')
    df = df.merge(book_stats, on='book_id', how='left')
    
    return df

# ПОДГОТОВКА ДАННЫХ
print("🔄 Подготовка тренировочных данных...")
train_labeled = train[train['has_read'] == 1].copy()
print(f"📚 Прочитанных книг для обучения: {len(train_labeled)}")

train_labeled = create_base_features(train_labeled, train_labeled)
train_labeled = train_labeled.merge(books, on='book_id', how='left')
train_labeled = train_labeled.merge(users, on='user_id', how='left')

print("🔄 Подготовка тестовых данных...")
test = create_base_features(test, train_labeled)
test = test.merge(books, on='book_id', how='left')
test = test.merge(users, on='user_id', how='left')

# ВЫБОР ПРИЗНАКОВ
feature_columns = [
    'user_mean_rating', 'user_rating_count',
    'book_mean_rating', 'book_rating_count',
    'gender', 'age', 'publication_year', 'language', 'avg_rating'
]

# Оставляем только существующие колонки
available_features = [col for col in feature_columns if col in train_labeled.columns]
print(f"🎯 Используется {len(available_features)} признаков: {available_features}")

# Заполнение пропусков
for col in available_features:
    train_labeled[col] = train_labeled[col].fillna(train_labeled[col].median())
    test[col] = test[col].fillna(test[col].median())

# КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ
cat_features = [col for col in ['gender', 'language'] if col in available_features]
print(f"📊 Категориальные признаки: {cat_features}")

# ОБУЧЕНИЕ МОДЕЛИ
print("\n🎓 ОБУЧЕНИЕ МОДЕЛИ...")
X_train = train_labeled[available_features]
y_train = train_labeled['rating']

print(f"Размер X_train: {X_train.shape}")
print(f"Размер y_train: {y_train.shape}")

# Простая модель для гарантированного запуска
model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_features,
    random_seed=42,
    verbose=50,
    early_stopping_rounds=20
)

model.fit(X_train, y_train)

# ПРЕДСКАЗАНИЕ
print("\n🔮 ПРЕДСКАЗАНИЕ ДЛЯ ТЕСТА...")
X_test = test[available_features]
test_predictions = model.predict(X_test)

# Ограничение предсказаний
test_predictions = np.clip(test_predictions, 0, 10)

# СОЗДАНИЕ САБМИТА
print("\n💾 СОЗДАНИЕ ФАЙЛА САБМИТА...")
submission = test[['user_id', 'book_id']].copy()
submission['rating_predict'] = test_predictions

submission_file = 'submission_simple.csv'
submission.to_csv(submission_file, index=False)

print(f"✅ САБМИТ СОХРАНЕН: {submission_file}")
print(f"📊 Размер сабмита: {submission.shape}")
print(f"📈 Диапазон предсказаний: {submission['rating_predict'].min():.2f} - {submission['rating_predict'].max():.2f}")

# ВАЛИДАЦИЯ
print("\n📊 ВАЛИДАЦИЯ НА ТРЕЙНЕ...")
train_predictions = model.predict(X_train)

rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
mae = mean_absolute_error(y_train, train_predictions)
score = 1 - (0.5 * (rmse/10) + 0.5 * (mae/10))

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}") 
print(f"SCORE: {score:.4f}")

print("\n🎉 РЕШЕНИЕ ЗАВЕРШЕНО! Файл submission_simple.csv готов для отправки.")