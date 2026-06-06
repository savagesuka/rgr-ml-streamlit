#& "C:\Program Files\Python311\python.exe" -m streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
# ==================== НАСТРОЙКА СТРАНИЦЫ ====================
st.set_page_config(page_title="ML РГР - Ушаков В.", page_icon="🎓", layout="wide")

# ==================== СТИЛИ ====================
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==================== ЗАГРУЗКА ДАННЫХ ====================
@st.cache_data
def load_data():
    df = pd.read_csv('ml_classification.csv')
    return df

# ==================== ЗАГРУЗКА МОДЕЛЕЙ ====================
@st.cache_resource
def load_models():
    models = {}
    
    # KNN
    with open('models/model_1_knn.pkl', 'rb') as f:
        models['knn'] = pickle.load(f)
    
    # Gradient Boosting
    with open('models/model_2_boosting.pkl', 'rb') as f:
        models['gradient_boosting'] = pickle.load(f)
    
    # LightGBM
    models['lightgbm'] = lgb.Booster(model_file='models/model_3_lightgbm.txt')
    
    # Bagging
    with open('models/model_4_bagging.pkl', 'rb') as f:
        models['bagging'] = pickle.load(f)
    
    # Stacking
    with open('models/model_5_stacking.pkl', 'rb') as f:
        models['stacking'] = pickle.load(f)
    
    # НЕЙРОСЕТЬ
    models['neural_network'] = load_model('models/model_6_neural_network.keras')
    
    return models

# ==================== ОБУЧЕНИЕ SCALER ====================
@st.cache_resource
def get_scaler():
    df = load_data()
    X = df.drop('Target', axis=1)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

# ==================== ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ ====================
def predict(model_name, input_data, scaler, models):
    input_scaled = scaler.transform(input_data)
    model = models[model_name]
    
    if model_name == 'lightgbm':
        result = model.predict(input_scaled)
    elif model_name == 'neural_network':
        probs = model.predict(input_scaled)
        result = np.argmax(probs, axis=1)
        #zxc=888
    else:
        result = model.predict(input_scaled)
    
    # Если одна строка — возвращаем число
    if input_data.shape[0] == 1:
        return result[0] if hasattr(result, '__len__') else result
    else:
        return result

# ==================== НАВИГАЦИЯ ====================
st.sidebar.markdown("# 🧭 Навигация")

page = st.sidebar.radio(
    "Выберите страницу:",
    ["Главная", "О разработчике", "О датасете", "Визуализации", "Предсказания"]
)

# ==================== СТРАНИЦА: ГЛАВНАЯ ====================
if page == "Главная":
    st.title("🎓 Data Engineering & ML - РГР")
    
    st.markdown("""
    ## 👨‍🎓 Ушаков Владислав | ФИТ-241
    
    ### Тема работы: Прогнозирование успеваемости студентов
    
    Данное web-приложение демонстрирует работу 5 моделей машинного обучения,
    обученных на датасете "Predict Students' Dropout and Academic Success".
    
    ### 🔍 Возможности приложения:
    
    - 📊 **Анализ датасета** — описание признаков и EDA
    - 📈 **Визуализации** — графики распределений и зависимостей
    - 🔮 **Предсказания** — выбор модели и ввод данных для прогноза
    
    ### 🤖 Используемые модели:
    - K-Nearest Neighbors (KNN)
    - Gradient Boosting
    - LightGBM
    - Bagging Classifier
    - Stacking Classifier
    - FCNN (Keras Tuner)
    """)
    
    st.divider()
    st.info("👈 **Используйте боковое меню для навигации по страницам**")

# ==================== СТРАНИЦА: О РАЗРАБОТЧИКЕ ====================
elif page == "О разработчике":
    st.title("👤 О разработчике")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("ava.jpg", width=150)
        st.caption("Фото студента")
    
    with col2:
        st.markdown("""
        ### Ушаков Владислав
        
        | Поле | Значение |
        |------|----------|
        | **ФИО** | Ушаков Владислав |
        | **Группа** | ФИТ-241 |
        | **Тема РГР** | Прогнозирование успеваемости студентов (классификация) |
        
        ### Стек технологий:
        - **Python** (pandas, numpy, sklearn)
        - **ML модели**: KNN, Gradient Boosting, LightGBM, Bagging, Stacking, FCNN
        - **Визуализация**: matplotlib, seaborn
        - **Web-фреймворк**: Streamlit
        """)

# ==================== СТРАНИЦА: О ДАТАСЕТЕ ====================
elif page == "О датасете":
    st.title("📊 О датасете")
    
    df = load_data()
    
    st.markdown("""
    ### 📁 Источник данных
    Датасет **"Predict Students' Dropout and Academic Success"** содержит информацию о студентах высшего учебного заведения.
    
    ### 🎯 Целевая переменная
    **Target** — статус студента:
    - **0** — Dropout (отчислен)
    - **1** — Graduate (выпустился)  
    - **2** — Enrolled (продолжает обучение)
    
    ### 📋 Признаки (37 штук)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **👤 Демографические:**
        - Marital status (Семейное положение)
        - Nacionality (Национальность)
        - Gender (Пол)
        
        **📚 Образование:**
        - Course (Курс)
        - Previous qualification (Предыдущая квалификация)
        - Previous qualification grade (Оценка)
        - Admission grade (Вступительный балл)
        
        **💰 Финансовые:**
        - Debtor (Должник)
        - Tuition fees up to date (Оплата обучения)
        - Scholarship holder (Стипендиат)
        """)
    
    with col2:
        st.markdown("""
        **📖 Успеваемость (1 семестр):**
        - Curricular units 1st sem (credited, enrolled, evaluations, approved, grade)
        
        **📖 Успеваемость (2 семестр):**
        - Curricular units 2nd sem (credited, enrolled, evaluations, approved, grade)
        
        **📈 Экономические:**
        - Unemployment rate
        - Inflation rate
        - GDP
        """)
    
    st.divider()
    
    st.subheader("📏 Размеры датасета")
    st.write(f"- Количество строк: **{df.shape[0]}**")
    st.write(f"- Количество столбцов: **{df.shape[1]}**")
    st.write(f"- Пропуски: **{df.isnull().sum().sum()}**")
    
    st.subheader("🏷️ Распределение целевой переменной")
    target_counts = df['Target'].value_counts().sort_index()
    target_labels = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(pd.DataFrame({
            'Статус': [target_labels[i] for i in target_counts.index],
            'Количество': target_counts.values,
            'Доля': [f"{val:.1f}%" for val in target_counts.values / len(df) * 100]
        }))
    
    with col2:
        fig, ax = plt.subplots()
        ax.pie(target_counts.values, labels=[target_labels[i] for i in target_counts.index], autopct='%1.1f%%')
        ax.set_title('Распределение Target')
        st.pyplot(fig)

# ==================== СТРАНИЦА: ВИЗУАЛИЗАЦИИ ====================
elif page == "Визуализации":
    st.title("📈 Визуализации данных")
    
    df = load_data()
    target_labels = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}
    df['Target_label'] = df['Target'].map(target_labels)
    
    # Визуализация 1
    st.subheader("1️⃣ Распределение возраста при поступлении")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x='Age at enrollment', hue='Target_label', bins=30, alpha=0.6, ax=ax1)
    ax1.set_title('Распределение возраста при поступлении по статусам студентов')
    st.pyplot(fig1)
    
    # Визуализация 2
    st.subheader("2️⃣ Средняя оценка за оба семестра")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='Target_label', y='Avg_grade_both_sem', ax=ax2)
    ax2.set_title('Распределение средних оценок по статусам студентов')
    st.pyplot(fig2)
    
    # Визуализация 3
    st.subheader("3️⃣ Корреляционная матрица (топ-15 признаков)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_corr = df[numeric_cols].corr()['Target'].abs().sort_values(ascending=False)
    top_features = target_corr.head(16).index.tolist()
    
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax3)
    ax3.set_title('Корреляционная матрица (топ-15 признаков)')
    st.pyplot(fig3)
    
    # Визуализация 4
    st.subheader("4️⃣ Семейное положение студентов")
    marital_map = {1: 'Single', 2: 'Married', 3: 'Widower', 4: 'Divorced', 5: 'Fact union', 6: 'Legally separated'}
    df_graduate = df[df['Target'] == 1].copy()
    df_graduate['Marital status_label'] = df_graduate['Marital status'].map(marital_map)
    
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df_graduate, x='Marital status_label', ax=ax4)
    ax4.set_title('Распределение семейного положения среди выпускников')
    plt.xticks(rotation=45)
    st.pyplot(fig4)

# ==================== СТРАНИЦА: ПРЕДСКАЗАНИЯ ====================
elif page == "Предсказания":
    st.title("🔮 Предсказание статуса студента")
    
    df = load_data()
    scaler = get_scaler()
    models = load_models()
    
    target_labels = {0: '❌ Dropout (Отчислен)', 1: '✅ Graduate (Выпустился)', 2: '📚 Enrolled (Продолжает)'}
    
    # Выбор модели
    model_options = {
        'knn': 'K-Nearest Neighbors (KNN)',
        'gradient_boosting': 'Gradient Boosting',
        'lightgbm': 'LightGBM',
        'bagging': 'Bagging Classifier',
        'stacking': 'Stacking Classifier',
        'neural_network': 'Нейронная сеть (Keras Tuner)'
    }
    
    selected_model = st.selectbox(
        "Выберите модель ML",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )
    
    st.markdown(f"**Выбрана модель:** `{model_options[selected_model]}`")
    
    input_method = st.radio(
        "Выберите способ ввода данных:",
        ["📁 Загрузить CSV-файл", "✍️ Ручной ввод"]
    )
    
    if input_method == "📁 Загрузить CSV-файл":
        uploaded_file = st.file_uploader("Загрузите CSV-файл с признаками", type=['csv'])
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            expected_columns = df.drop('Target', axis=1).columns
            missing_cols = set(expected_columns) - set(input_df.columns)
            
            if missing_cols:
                st.error(f"❌ В файле отсутствуют колонки: {missing_cols}")
            else:
                input_df = input_df[expected_columns]
                
                if st.button("🔍 Выполнить предсказание"):
                    predictions = predict(selected_model, input_df, scaler, models)
                    results_df = input_df.copy()
                    results_df['Predicted_Target'] = predictions
                    results_df['Predicted_Status'] = results_df['Predicted_Target'].map(target_labels)
                    
                    st.success("✅ Предсказания выполнены!")
                    st.dataframe(results_df[['Predicted_Status'] + list(input_df.columns[:5])])
    
    else:  # Ручной ввод
        st.markdown("### Введите основные признаки студента")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Возраст при поступлении", min_value=17, max_value=100, value=20)
            admission_grade = st.number_input("Вступительный балл", min_value=0.0, max_value=200.0, value=120.0)
            scholarship = st.selectbox("Стипендиат", [0, 1], format_func=lambda x: 'Нет' if x == 0 else 'Да')
        
        with col2:
            prev_grade = st.number_input("Оценка предыдущей квалификации", min_value=0.0, max_value=200.0, value=130.0)
            unemployment = st.number_input("Уровень безработицы (%)", min_value=5.0, max_value=20.0, value=10.8)
        
        if st.button("🎯 Предсказать статус", type="primary"):
            feature_names = df.drop('Target', axis=1).columns
            input_dict = {}
            
            for col in feature_names:
                if col == 'Age at enrollment':
                    input_dict[col] = age
                elif col == 'Admission grade':
                    input_dict[col] = admission_grade
                elif col == 'Scholarship holder':
                    input_dict[col] = scholarship
                elif col == 'Previous qualification (grade)':
                    input_dict[col] = prev_grade
                elif col == 'Unemployment rate':
                    input_dict[col] = unemployment
                else:
                    input_dict[col] = df[col].median()
            
            input_df = pd.DataFrame([input_dict])
            prediction = predict(selected_model, input_df, scaler, models)
            
            st.divider()
            st.markdown("## 📋 Результат предсказания:")
            
            if prediction == 1:
                st.success(f"### ✅ Студент, скорее всего, **ВЫПУСТИТСЯ**")
                st.balloons()
            elif prediction == 0:
                st.error(f"### ❌ Студент, скорее всего, будет **ОТЧИСЛЕН**")
            else:
                st.warning(f"### 📚 Студент, скорее всего, **ПРОДОЛЖИТ ОБУЧЕНИЕ**")
            
            st.markdown(f"**Статус:** `{target_labels[prediction]}`")