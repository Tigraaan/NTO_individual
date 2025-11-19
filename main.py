def run_enhanced_pipeline():
    """Запуск улучшенного пайплайна"""
    print("=== УЛУЧШЕННЫЙ ПАЙПЛАЙН ===")
    
    # Загрузка данных
    predictor = EnhancedBookRatingPredictor()
    predictor.load_data()
    
    # Анализ данных
    explore_data(predictor.train, predictor.books, predictor.users)
    
    # Подготовка улучшенных признаков
    X, y = predictor.prepare_train_data()
    X_enhanced = predictor.prepare_enhanced_features(predictor.train[predictor.train['has_read'] == 1])
    
    print(f"Улучшенные признаки: {X_enhanced.shape[1]}")
    
    # Обучение улучшенного ансамбля
    print("\nОбучение улучшенного ансамбля...")
    ensemble = create_advanced_ensemble()
    
    # Кросс-валидация
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(ensemble, X_enhanced, y, scoring='neg_mean_squared_error', cv=3)
    rmse_scores = np.sqrt(-scores)
    print(f"CV RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
    
    # Обучение на всех данных
    ensemble.fit(X_enhanced, y)
    predictor.models['ensemble'] = ensemble
    
    # Предсказание для теста
    X_test_enhanced = predictor.prepare_enhanced_features(predictor.test, is_train=False)
    predictions = ensemble.predict(X_test_enhanced)
    predictions = np.clip(predictions, 0, 10)
    
    # Создание сабмита
    submission = predictor.test[['user_id', 'book_id']].copy()
    submission['rating_predict'] = predictions
    submission.to_csv('submission_enhanced.csv', index=False)
    
    print(f"\nУлучшенный сабмит создан!")
    print(f"Статистика предсказаний: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
    
    return submission, ensemble

# Запуск улучшенной версии
if __name__ == "__main__":
    submission, model = run_enhanced_pipeline()