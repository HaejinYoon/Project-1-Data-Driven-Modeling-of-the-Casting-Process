#베이스 모델 1

# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings

# # XGBoost와 LightGBM
# try:
#     import xgboost as xgb
#     XGBOOST_AVAILABLE = True
# except ImportError:
#     XGBOOST_AVAILABLE = False

# try:
#     import lightgbm as lgb
#     LIGHTGBM_AVAILABLE = True
# except ImportError:
#     LIGHTGBM_AVAILABLE = False

# warnings.filterwarnings('ignore')

# # 한글 폰트 설정
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False

# print("🎯 불량률 예측 중심 베이스라인 모델")
# print("="*60)

# # 1. 데이터 로딩 및 기본 정보 확인
# print("📁 원본 데이터 로딩...")
# train_raw = pd.read_csv('./data/train.csv')

# print(f"✅ 원본 데이터: {train_raw.shape}")
# print(f"📊 원본 클래스 분포:")

# if 'passorfail' in train_raw.columns:
#     original_counts = train_raw['passorfail'].value_counts().sort_index()
#     total_count = len(train_raw)
#     good_count = original_counts.get(0, 0)
#     defect_count = original_counts.get(1, 0)
#     defect_rate = defect_count / total_count * 100
    
#     print(f"   📈 전체: {total_count:,}개")
#     print(f"   ✅ 양품(0): {good_count:,}개 ({good_count/total_count*100:.1f}%)")
#     print(f"   ❌ 불량품(1): {defect_count:,}개 ({defect_count/total_count*100:.1f}%)")
#     print(f"   🎯 원본 불량률: {defect_rate:.2f}%")
# else:
#     print("❌ 'passorfail' 컬럼을 찾을 수 없습니다!")
#     exit()

# # 2. 🎲 전체 데이터를 잘 섞어서 train/test 분할 (8:2)
# print(f"\n🎲 전체 데이터 섞어서 train/test 분할 (8:2)...")

# # 층화 추출로 불량률 유지하며 분할
# X_raw = train_raw.drop('passorfail', axis=1)
# y_raw = train_raw['passorfail']

# X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
#     X_raw, y_raw, 
#     test_size=0.2, 
#     random_state=42, 
#     stratify=y_raw,
#     shuffle=True  # 잘 섞기
# )

# print(f"✅ 분할 완료:")
# print(f"   🔧 Train: {len(X_train_raw):,}개")
# train_defect_rate = (y_train_raw == 1).sum() / len(y_train_raw) * 100
# print(f"      불량률: {train_defect_rate:.2f}% ({(y_train_raw == 1).sum():,}개)")

# print(f"   🧪 Test: {len(X_test_raw):,}개") 
# test_defect_rate = (y_test_raw == 1).sum() / len(y_test_raw) * 100
# print(f"      불량률: {test_defect_rate:.2f}% ({(y_test_raw == 1).sum():,}개)")

# # 3. 전처리 함수 정의
# def preprocess_data(X, data_name):
#     """데이터 전처리"""
#     print(f"\n🔧 {data_name} 전처리...")
#     X_processed = X.copy()
    
#     print(f"   📊 전처리 전: {X_processed.shape}")
    
#     # 불필요한 컬럼 제거
#     drop_columns = ['id', 'line', 'name', 'mold_name', 'registration_time', 'time', 'date']
#     existing_drop_columns = [col for col in drop_columns if col in X_processed.columns]
    
#     if existing_drop_columns:
#         print(f"   🗑️ 제거할 컬럼: {existing_drop_columns}")
#         X_processed = X_processed.drop(columns=existing_drop_columns)
    
#     # 결측치 확인
#     missing_info = X_processed.isnull().sum()
#     missing_cols = missing_info[missing_info > 0].head(10)
#     if len(missing_cols) > 0:
#         print(f"   🔍 주요 결측치:")
#         for col, count in missing_cols.items():
#             print(f"      {col}: {count:,}개 ({count/len(X_processed)*100:.1f}%)")
    
#     # 특정 컬럼 결측치 처리
#     if 'heating_furnace' in X_processed.columns:
#         before = X_processed['heating_furnace'].isnull().sum()
#         X_processed['heating_furnace'].fillna('C', inplace=True)
#         print(f"      ✅ heating_furnace: {before}개 → 'C'로 대체")
    
#     if 'tryshot_signal' in X_processed.columns:
#         before = X_processed['tryshot_signal'].isnull().sum()
#         X_processed['tryshot_signal'].fillna('0', inplace=True)
#         print(f"      ✅ tryshot_signal: {before}개 → '0'으로 대체")
    
#     # 남은 결측치 제거
#     before_drop = len(X_processed)
#     X_processed = X_processed.dropna()
#     after_drop = len(X_processed)
#     dropped = before_drop - after_drop
    
#     if dropped > 0:
#         print(f"   ⚠️ dropna로 {dropped:,}행 제거 ({dropped/before_drop*100:.1f}%)")
    
#     # 날짜/시간 컬럼 추가 제거
#     datetime_cols = []
#     for col in X_processed.columns:
#         if X_processed[col].dtype == 'object':
#             sample_vals = X_processed[col].dropna().head(3).astype(str).tolist()
#             has_date_pattern = any(
#                 len(val) >= 8 and ('-' in val or '/' in val or ':' in val)
#                 for val in sample_vals
#             )
#             if has_date_pattern:
#                 datetime_cols.append(col)
    
#     if datetime_cols:
#         print(f"   🗑️ 날짜/시간 컬럼 제거: {datetime_cols}")
#         X_processed = X_processed.drop(columns=datetime_cols)
    
#     # 범주형 변수 원핫 인코딩
#     categorical_cols = []
#     for col in X_processed.columns:
#         if X_processed[col].dtype == 'object':
#             unique_count = X_processed[col].nunique()
#             if unique_count <= 50:  # 50개 이하만 인코딩
#                 categorical_cols.append(col)
#             else:
#                 print(f"   ⚠️ {col}: 고유값 {unique_count}개로 인코딩 제외")
    
#     if categorical_cols:
#         print(f"   🏷️ 원핫 인코딩: {categorical_cols}")
#         X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
    
#     # 비수치형 컬럼 제거
#     non_numeric = X_processed.select_dtypes(exclude=['number']).columns.tolist()
#     if non_numeric:
#         print(f"   🗑️ 비수치형 컬럼 제거: {non_numeric}")
#         X_processed = X_processed.drop(columns=non_numeric)
    
#     print(f"   ✅ 전처리 완료: {X_processed.shape}")
#     return X_processed

# # 4. Train/Test 각각 전처리
# X_train_processed = preprocess_data(X_train_raw, "Train")
# X_test_processed = preprocess_data(X_test_raw, "Test")

# # 전처리 후 실제 남은 데이터의 y 값들 (인덱스 기준으로 매칭)
# y_train_processed = y_train_raw.loc[X_train_processed.index]
# y_test_processed = y_test_raw.loc[X_test_processed.index]

# print(f"\n📊 전처리 후 최종 데이터:")
# print(f"   🔧 Train: {X_train_processed.shape}, 불량률: {(y_train_processed==1).sum()/len(y_train_processed)*100:.2f}%")
# print(f"   🧪 Test: {X_test_processed.shape}, 불량률: {(y_test_processed==1).sum()/len(y_test_processed)*100:.2f}%")

# # 5. 컬럼 통일
# print(f"\n🔄 Train/Test 컬럼 통일...")
# train_cols = set(X_train_processed.columns)
# test_cols = set(X_test_processed.columns)

# # 공통 컬럼만 사용
# common_cols = list(train_cols & test_cols)
# print(f"   📊 공통 컬럼: {len(common_cols)}개")

# X_train_final = X_train_processed[common_cols]
# X_test_final = X_test_processed[common_cols]

# print(f"   ✅ 컬럼 통일 완료: Train {X_train_final.shape}, Test {X_test_final.shape}")

# # 6. 🎯 클래스 불균형 해결 (Train만)
# print(f"\n⚖️ 클래스 불균형 해결 (Train만)...")

# train_counts = y_train_processed.value_counts().sort_index()
# good_count = train_counts.get(0, 0)
# defect_count = train_counts.get(1, 0)

# print(f"   📊 Train 원본 분포:")
# print(f"      양품: {good_count:,}개, 불량품: {defect_count:,}개")
# print(f"      불량률: {defect_count/(good_count+defect_count)*100:.2f}%")

# # 언더샘플링 (6:4 비율로 조정)
# target_ratio = 0.4  # 불량품 비율을 40%로
# target_defect_count = defect_count
# target_good_count = int(defect_count / target_ratio * (1 - target_ratio))

# print(f"   🎯 목표 분포 (불량품 {target_ratio*100:.0f}%):")
# print(f"      양품: {target_good_count:,}개, 불량품: {target_defect_count:,}개")

# # 언더샘플링 실행
# good_indices = y_train_processed[y_train_processed == 0].index
# defect_indices = y_train_processed[y_train_processed == 1].index

# np.random.seed(42)
# sampled_good_indices = np.random.choice(good_indices, target_good_count, replace=False)
# final_indices = np.concatenate([sampled_good_indices, defect_indices])
# np.random.shuffle(final_indices)

# X_train_balanced = X_train_final.loc[final_indices]
# y_train_balanced = y_train_processed.loc[final_indices]

# balanced_counts = y_train_balanced.value_counts().sort_index()
# print(f"   ✅ 균형 조정 완료:")
# print(f"      양품: {balanced_counts.get(0, 0):,}개, 불량품: {balanced_counts.get(1, 0):,}개")
# print(f"      불량률: {balanced_counts.get(1, 0)/len(y_train_balanced)*100:.2f}%")

# # 7. 베이스라인 모델들 정의
# print(f"\n🤖 베이스라인 모델 정의...")

# models = {
#     'Random Forest': RandomForestClassifier(
#         n_estimators=100,
#         max_depth=10,
#         random_state=42,
#         n_jobs=-1,
#         class_weight='balanced'
#     ),
#     'Gradient Boosting': GradientBoostingClassifier(
#         n_estimators=100,
#         learning_rate=0.1,
#         max_depth=6,
#         random_state=42
#     ),
#     'Logistic Regression': LogisticRegression(
#         random_state=42,
#         max_iter=1000,
#         class_weight='balanced'
#     )
# }

# if XGBOOST_AVAILABLE:
#     models['XGBoost'] = xgb.XGBClassifier(
#         n_estimators=100,
#         learning_rate=0.1,
#         max_depth=6,
#         random_state=42,
#         eval_metric='logloss'
#     )

# if LIGHTGBM_AVAILABLE:
#     models['LightGBM'] = lgb.LGBMClassifier(
#         n_estimators=100,
#         learning_rate=0.1,
#         max_depth=6,
#         random_state=42,
#         verbose=-1,
#         class_weight='balanced'
#     )

# print(f"   📊 총 {len(models)}개 모델 준비")

# # 8. 스케일링
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_balanced)
# X_test_scaled = scaler.transform(X_test_final)

# scaling_models = ['Logistic Regression']

# # 9. 모델 학습 및 평가
# print(f"\n🏋️ 모델 학습 및 평가...")
# results = {}

# for name, model in models.items():
#     print(f"\n🔄 {name} 학습...")
    
#     # 스케일링 데이터 선택
#     if name in scaling_models:
#         X_train_use = X_train_scaled
#         X_test_use = X_test_scaled
#     else:
#         X_train_use = X_train_balanced
#         X_test_use = X_test_final
    
#     # 학습
#     model.fit(X_train_use, y_train_balanced)
    
#     # 예측
#     test_pred = model.predict(X_test_use)
#     test_proba = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
    
#     # 성능 계산
#     accuracy = accuracy_score(y_test_processed, test_pred)
#     precision = precision_score(y_test_processed, test_pred)
#     recall = recall_score(y_test_processed, test_pred)
#     f1 = f1_score(y_test_processed, test_pred)
    
#     # 🎯 핵심: 불량률 예측 정확도
#     actual_defect_rate = (y_test_processed == 1).sum() / len(y_test_processed) * 100
#     predicted_defect_rate = (test_pred == 1).sum() / len(test_pred) * 100
#     defect_rate_error = abs(actual_defect_rate - predicted_defect_rate)
    
#     # 교차검증
#     cv_scores = cross_val_score(model, X_train_use, y_train_balanced, cv=5, scoring='accuracy')
    
#     results[name] = {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'cv_mean': cv_scores.mean(),
#         'cv_std': cv_scores.std(),
#         'actual_defect_rate': actual_defect_rate,
#         'predicted_defect_rate': predicted_defect_rate,
#         'defect_rate_error': defect_rate_error,
#         'predictions': test_pred,
#         'probabilities': test_proba
#     }
    
#     print(f"   ✅ Accuracy: {accuracy:.4f}")
#     print(f"   ✅ Precision: {precision:.4f}")
#     print(f"   ✅ Recall: {recall:.4f}")
#     print(f"   ✅ F1-Score: {f1:.4f}")
#     print(f"   🎯 실제 불량률: {actual_defect_rate:.2f}%")
#     print(f"   🎯 예측 불량률: {predicted_defect_rate:.2f}%")
#     print(f"   🎯 불량률 오차: {defect_rate_error:.2f}%p")
#     print(f"   📊 CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# # 10. 결과 비교 및 시각화
# print(f"\n📊 모델 성능 비교...")

# results_df = pd.DataFrame(results).T.round(4)
# results_df = results_df.sort_values('defect_rate_error', ascending=True)  # 불량률 오차 기준 정렬

# print(f"\n🏆 불량률 예측 정확도 순위:")
# print("="*90)
# print(f"{'순위':>2} {'모델':15} {'정확도':>8} {'정밀도':>8} {'재현율':>8} {'F1':>6} {'실제불량률':>8} {'예측불량률':>8} {'오차':>6}")
# print("-"*90)

# for i, (model, row) in enumerate(results_df.iterrows(), 1):
#     print(f"{i:2d}. {model:15s} "
#           f"{row['accuracy']:8.4f} "
#           f"{row['precision']:8.4f} "
#           f"{row['recall']:8.4f} "
#           f"{row['f1']:6.4f} "
#           f"{row['actual_defect_rate']:7.2f}% "
#           f"{row['predicted_defect_rate']:7.2f}% "
#           f"{row['defect_rate_error']:5.2f}%p")

# # 11. 상세 시각화
# fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# models_list = list(results.keys())

# # 11-1. 불량률 예측 정확도
# ax1 = axes[0, 0]
# actual_rates = [results[m]['actual_defect_rate'] for m in models_list]
# predicted_rates = [results[m]['predicted_defect_rate'] for m in models_list]

# x = np.arange(len(models_list))
# width = 0.35

# ax1.bar(x - width/2, actual_rates, width, label='실제 불량률', alpha=0.8, color='red')
# ax1.bar(x + width/2, predicted_rates, width, label='예측 불량률', alpha=0.8, color='blue')
# ax1.set_xlabel('모델')
# ax1.set_ylabel('불량률 (%)')
# ax1.set_title('실제 vs 예측 불량률 비교', fontweight='bold')
# ax1.set_xticks(x)
# ax1.set_xticklabels(models_list, rotation=45)
# ax1.legend()
# ax1.grid(axis='y', alpha=0.3)

# # 11-2. 불량률 오차
# ax2 = axes[0, 1]
# errors = [results[m]['defect_rate_error'] for m in models_list]
# colors = ['green' if e <= 1 else 'orange' if e <= 2 else 'red' for e in errors]

# bars = ax2.bar(range(len(models_list)), errors, color=colors, alpha=0.8)
# ax2.set_xlabel('모델')
# ax2.set_ylabel('불량률 오차 (%p)')
# ax2.set_title('불량률 예측 오차', fontweight='bold')
# ax2.set_xticks(range(len(models_list)))
# ax2.set_xticklabels(models_list, rotation=45)
# ax2.grid(axis='y', alpha=0.3)

# # 기준선 추가
# ax2.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='우수 (1%p)')
# ax2.axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='보통 (2%p)')
# ax2.legend()

# # 11-3. 정확도 vs 불량률 오차 스캐터
# ax3 = axes[0, 2]
# accuracies = [results[m]['accuracy'] for m in models_list]
# scatter = ax3.scatter(errors, accuracies, s=100, alpha=0.7)

# for i, model in enumerate(models_list):
#     ax3.annotate(model, (errors[i], accuracies[i]), 
#                 xytext=(5, 5), textcoords='offset points', fontsize=8)

# ax3.set_xlabel('불량률 오차 (%p)')
# ax3.set_ylabel('정확도')
# ax3.set_title('정확도 vs 불량률 예측 정확도', fontweight='bold')
# ax3.grid(True, alpha=0.3)

# # 11-4. 최고 모델의 혼동행렬
# best_model = results_df.index[0]
# ax4 = axes[1, 0]

# cm = confusion_matrix(y_test_processed, results[best_model]['predictions'])
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
#             xticklabels=['양품(0)', '불량품(1)'], yticklabels=['양품(0)', '불량품(1)'])
# ax4.set_title(f'{best_model} - 혼동행렬', fontweight='bold')
# ax4.set_xlabel('예측')
# ax4.set_ylabel('실제')

# # 11-5. 모델별 F1 점수
# ax5 = axes[1, 1]
# f1_scores = [results[m]['f1'] for m in models_list]
# bars = ax5.bar(range(len(models_list)), f1_scores, color='lightgreen', alpha=0.8)
# ax5.set_xlabel('모델')
# ax5.set_ylabel('F1 Score')
# ax5.set_title('모델별 F1 점수', fontweight='bold')
# ax5.set_xticks(range(len(models_list)))
# ax5.set_xticklabels(models_list, rotation=45)
# ax5.grid(axis='y', alpha=0.3)

# # 11-6. 교차검증 점수
# ax6 = axes[1, 2]
# cv_means = [results[m]['cv_mean'] for m in models_list]
# cv_stds = [results[m]['cv_std'] for m in models_list]

# bars = ax6.bar(range(len(models_list)), cv_means, yerr=cv_stds,
#                color='gold', alpha=0.8, capsize=5)
# ax6.set_xlabel('모델')
# ax6.set_ylabel('CV Score')
# ax6.set_title('교차검증 점수', fontweight='bold')
# ax6.set_xticks(range(len(models_list)))
# ax6.set_xticklabels(models_list, rotation=45)
# ax6.grid(axis='y', alpha=0.3)

# plt.tight_layout()
# plt.suptitle('🎯 불량률 예측 중심 베이스라인 모델 비교', fontsize=16, fontweight='bold', y=0.98)
# plt.show()

# # 12. 최종 요약 및 추천
# print(f"\n🎉 베이스라인 모델 분석 완료!")
# print("="*60)

# best_defect_model = results_df.index[0]
# best_accuracy_model = results_df.sort_values('accuracy', ascending=False).index[0]

# print(f"🏆 불량률 예측 최고 모델: {best_defect_model}")
# print(f"   🎯 불량률 오차: {results[best_defect_model]['defect_rate_error']:.2f}%p")
# print(f"   📊 정확도: {results[best_defect_model]['accuracy']:.4f}")

# if best_accuracy_model != best_defect_model:
#     print(f"🥇 정확도 최고 모델: {best_accuracy_model}")
#     print(f"   📊 정확도: {results[best_accuracy_model]['accuracy']:.4f}")
#     print(f"   🎯 불량률 오차: {results[best_accuracy_model]['defect_rate_error']:.2f}%p")

# print(f"\n📊 전체 요약:")
# print(f"   📈 원본 데이터: {train_raw.shape[0]:,}개 (불량률 {defect_rate:.2f}%)")
# print(f"   🔧 Train 최종: {len(y_train_balanced):,}개 (불량률 {balanced_counts.get(1,0)/len(y_train_balanced)*100:.2f}%)")
# print(f"   🧪 Test 최종: {len(y_test_processed):,}개 (불량률 {actual_defect_rate:.2f}%)")
# print(f"   🎯 최고 불량률 예측 정확도: {results_df.iloc[0]['defect_rate_error']:.2f}%p 오차")

# print(f"\n💾 저장된 주요 변수:")
# print(f"   - models: 학습된 모델들")
# print(f"   - results: 각 모델의 상세 결과")
# print(f"   - X_test_final, y_test_processed: 테스트 데이터")
# print(f"   - scaler: 표준화 스케일러")

# print(f"\n✅ 베이스라인 모델 완료! 🎯")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# XGBoost, LightGBM 체크
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("🎯 불균형 해결 방법 비교 분석")
print("="*60)

# 1. 데이터 로딩
train_raw = pd.read_csv('./data/train.csv')
print(f"✅ 원본 데이터: {train_raw.shape}")

if 'passorfail' in train_raw.columns:
    original_counts = train_raw['passorfail'].value_counts().sort_index()
    total_count = len(train_raw)
    good_count = original_counts.get(0, 0)
    defect_count = original_counts.get(1, 0)
    defect_rate = defect_count / total_count * 100
    print(f"📊 원본 클래스 분포:")
    print(f"   ✅ 양품(0): {good_count:,}개 ({good_count/total_count*100:.1f}%)")
    print(f"   ❌ 불량품(1): {defect_count:,}개 ({defect_count/total_count*100:.1f}%)")
    print(f"   🎯 원본 불량률: {defect_rate:.2f}%")

# 2. Train/Test Split
X_raw = train_raw.drop('passorfail', axis=1)
y_raw = train_raw['passorfail']

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw,
    test_size=0.2, random_state=42,
    stratify=y_raw, shuffle=True
)

print(f"   🔧 Train: {len(X_train_raw):,}개, Test: {len(X_test_raw):,}개")

# 3. 전처리 함수 (NaN → 최빈값)
def preprocess_data(X, data_name):
    print(f"\n🔧 {data_name} 전처리...")
    X_processed = X.copy()

    # 불필요한 컬럼 제거
    drop_columns = ['id', 'line', 'name', 'mold_name', 'registration_time', 'time', 'date']
    existing_drop_columns = [col for col in drop_columns if col in X_processed.columns]
    if existing_drop_columns:
        X_processed = X_processed.drop(columns=existing_drop_columns)

    # 특수 컬럼 채우기
    if 'heating_furnace' in X_processed.columns:
        X_processed['heating_furnace'].fillna('C', inplace=True)
    if 'tryshot_signal' in X_processed.columns:
        X_processed['tryshot_signal'].fillna('0', inplace=True)

    # NaN → 최빈값으로 채우기
    imputer = SimpleImputer(strategy='most_frequent')
    X_processed[:] = imputer.fit_transform(X_processed)

    # 날짜/시간 패턴 제거
    datetime_cols = []
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            sample_vals = X_processed[col].dropna().head(3).astype(str).tolist()
            has_date_pattern = any(
                len(val) >= 8 and ('-' in val or '/' in val or ':') in val
                for val in sample_vals
            )
            if has_date_pattern:
                datetime_cols.append(col)
    if datetime_cols:
        X_processed = X_processed.drop(columns=datetime_cols)

    # 범주형 → 원핫 인코딩
    categorical_cols = [
        col for col in X_processed.columns
        if X_processed[col].dtype == 'object' and X_processed[col].nunique() <= 50
    ]
    if categorical_cols:
        X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)

    # 비수치형 제거
    non_numeric = X_processed.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric:
        X_processed = X_processed.drop(columns=non_numeric)

    print(f"   ✅ 전처리 완료: {X_processed.shape}")
    return X_processed

# 4. 전처리 실행
X_train_processed = preprocess_data(X_train_raw, "Train")
X_test_processed = preprocess_data(X_test_raw, "Test")
y_train_processed = y_train_raw.loc[X_train_processed.index]
y_test_processed = y_test_raw.loc[X_test_processed.index]

# 5. 컬럼 통일
common_cols = list(set(X_train_processed.columns) & set(X_test_processed.columns))
X_train_final = X_train_processed[common_cols]
X_test_final = X_test_processed[common_cols]

print(f"\n📊 전처리 후 최종 데이터:")
print(f"   Train: {X_train_final.shape}, Test: {X_test_final.shape}")

# ⚖️ 이하 모델 학습/평가 부분은 그대로 사용 가능

# 6. 📊 불균형 해결 방법들 정의
print(f"\n⚖️ 불균형 해결 방법 정의...")

# 6-1. 샘플링 기법들
def undersample_data(X, y, ratio=0.4, random_state=42):
    """언더샘플링"""
    np.random.seed(random_state)
    
    good_indices = y[y == 0].index
    defect_indices = y[y == 1].index
    
    defect_count = len(defect_indices)
    target_good_count = int(defect_count / ratio * (1 - ratio))
    
    sampled_good_indices = np.random.choice(good_indices, min(target_good_count, len(good_indices)), replace=False)
    final_indices = np.concatenate([sampled_good_indices, defect_indices])
    np.random.shuffle(final_indices)
    
    return X.loc[final_indices], y.loc[final_indices]

def oversample_data(X, y, ratio=0.4, random_state=42):
    """오버샘플링"""
    np.random.seed(random_state)
    
    good_indices = y[y == 0].index
    defect_indices = y[y == 1].index
    
    good_count = len(good_indices)
    target_defect_count = int(good_count * ratio / (1 - ratio))
    
    # 부족한 만큼 복제
    additional_defect = target_defect_count - len(defect_indices)
    if additional_defect > 0:
        sampled_defect_indices = np.random.choice(defect_indices, additional_defect, replace=True)
        all_defect_indices = np.concatenate([defect_indices, sampled_defect_indices])
    else:
        all_defect_indices = defect_indices
    
    final_indices = np.concatenate([good_indices, all_defect_indices])
    np.random.shuffle(final_indices)
    
    return X.loc[final_indices], y.loc[final_indices]

def mixed_sample_data(X, y, ratio=0.4, random_state=42):
    """혼합 샘플링 (언더+오버)"""
    np.random.seed(random_state)
    
    good_indices = y[y == 0].index
    defect_indices = y[y == 1].index
    
    good_count = len(good_indices)
    defect_count = len(defect_indices)
    
    # 목표: 전체의 80% 크기로 축소하면서 비율 맞추기
    target_total = int((good_count + defect_count) * 0.8)
    target_defect = int(target_total * ratio)
    target_good = target_total - target_defect
    
    # 양품 언더샘플링
    sampled_good_indices = np.random.choice(good_indices, min(target_good, good_count), replace=False)
    
    # 불량품 오버샘플링
    if target_defect > defect_count:
        additional_defect = target_defect - defect_count
        sampled_additional_defect = np.random.choice(defect_indices, additional_defect, replace=True)
        all_defect_indices = np.concatenate([defect_indices, sampled_additional_defect])
    else:
        all_defect_indices = np.random.choice(defect_indices, target_defect, replace=False)
    
    final_indices = np.concatenate([sampled_good_indices, all_defect_indices])
    np.random.shuffle(final_indices)
    
    return X.loc[final_indices], y.loc[final_indices]

def synthetic_sample_data(X, y, ratio=0.4, random_state=42):
    """합성 샘플 생성 (간단한 SMOTE)"""
    np.random.seed(random_state)
    
    good_data = X[y == 0]
    defect_data = X[y == 1]
    
    good_count = len(good_data)
    target_defect_count = int(good_count * ratio / (1 - ratio))
    additional_defect = target_defect_count - len(defect_data)
    
    if additional_defect > 0:
        # 합성 샘플 생성
        synthetic_samples = []
        for _ in range(additional_defect):
            # 랜덤하게 두 불량품 샘플 선택
            idx1, idx2 = np.random.choice(len(defect_data), 2, replace=True)
            sample1 = defect_data.iloc[idx1].values
            sample2 = defect_data.iloc[idx2].values
            
            # 선형 보간
            alpha = np.random.random()
            synthetic_sample = alpha * sample1 + (1 - alpha) * sample2
            synthetic_samples.append(synthetic_sample)
        
        # 합성 데이터 추가
        synthetic_df = pd.DataFrame(synthetic_samples, columns=X.columns)
        X_synthetic = pd.concat([good_data, defect_data, synthetic_df], ignore_index=True)
        y_synthetic = pd.concat([
            pd.Series([0] * len(good_data)),
            pd.Series([1] * len(defect_data)),
            pd.Series([1] * len(synthetic_samples))
        ], ignore_index=True)
        
        # 섞기
        shuffle_indices = np.random.permutation(len(X_synthetic))
        return X_synthetic.iloc[shuffle_indices], y_synthetic.iloc[shuffle_indices]
    else:
        return X, y

# 6-2. 가중치 계산 함수
def compute_sample_weights(y, method='balanced'):
    """샘플 가중치 계산"""
    if method == 'balanced':
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = {0: class_weights[0], 1: class_weights[1]}
        return np.array([weight_dict[label] for label in y])
    elif method == 'custom':
        # 불량품에 더 높은 가중치 (양품:불량품 = 1:3)
        return np.array([1 if label == 0 else 3 for label in y])
    else:
        return None

# 7. 📊 다양한 불균형 해결 방법 적용
print(f"\n🎯 다양한 불균형 해결 방법 테스트...")

# 7-1. 데이터 준비
methods_data = {}

# 원본 데이터 (불균형 그대로)
methods_data['Original'] = {
    'X': X_train_final.copy(),
    'y': y_train_processed.copy(),
    'description': '원본 불균형 데이터'
}

# 언더샘플링
X_under, y_under = undersample_data(X_train_final, y_train_processed, ratio=0.4)
methods_data['Undersample'] = {
    'X': X_under,
    'y': y_under,
    'description': '언더샘플링 (4:6 비율)'
}

# 오버샘플링
X_over, y_over = oversample_data(X_train_final, y_train_processed, ratio=0.4)
methods_data['Oversample'] = {
    'X': X_over,
    'y': y_over,
    'description': '오버샘플링 (4:6 비율)'
}

# 혼합 샘플링
X_mixed, y_mixed = mixed_sample_data(X_train_final, y_train_processed, ratio=0.4)
methods_data['Mixed'] = {
    'X': X_mixed,
    'y': y_mixed,
    'description': '혼합 샘플링 (언더+오버)'
}

# 합성 샘플
X_synthetic, y_synthetic = synthetic_sample_data(X_train_final, y_train_processed, ratio=0.4)
methods_data['Synthetic'] = {
    'X': X_synthetic,
    'y': y_synthetic,
    'description': '합성 샘플 생성 (SMOTE 유사)'
}

# 샘플 가중치 (원본 데이터 + 가중치)
sample_weights_balanced = compute_sample_weights(y_train_processed, 'balanced')
sample_weights_custom = compute_sample_weights(y_train_processed, 'custom')

methods_data['Weighted_Balanced'] = {
    'X': X_train_final.copy(),
    'y': y_train_processed.copy(),
    'sample_weight': sample_weights_balanced,
    'description': '원본 + 균형 가중치'
}

methods_data['Weighted_Custom'] = {
    'X': X_train_final.copy(),
    'y': y_train_processed.copy(),
    'sample_weight': sample_weights_custom,
    'description': '원본 + 불량품 3배 가중치'
}

# 각 방법별 데이터 분포 출력
print(f"\n📊 불균형 해결 방법별 데이터 분포:")
print("-" * 70)
for method, data in methods_data.items():
    y_counts = data['y'].value_counts().sort_index()
    total = len(data['y'])
    good_pct = y_counts.get(0, 0) / total * 100
    defect_pct = y_counts.get(1, 0) / total * 100
    
    weight_info = ""
    if 'sample_weight' in data:
        weight_info = " (가중치 적용)"
    
    print(f"{method:15s}: {total:6,}개 | 양품: {y_counts.get(0, 0):5,}({good_pct:4.1f}%) | "
          f"불량품: {y_counts.get(1, 0):5,}({defect_pct:4.1f}%){weight_info}")

# 8. 📊 모델 정의 (불균형 대응 옵션 포함)
print(f"\n🤖 모델 정의...")

def get_models():
    """다양한 불균형 대응 옵션을 가진 모델들"""
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'RandomForest_Balanced': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced'
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42, max_iter=1000
        ),
        'LogisticRegression_Balanced': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        )
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss'
        )
        models['XGBoost_Balanced'] = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss'
        )
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1
        )
        models['LightGBM_Balanced'] = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1, class_weight='balanced'
        )
    
    return models

# 9. 🏋️ 대규모 실험 실행
print(f"\n🏋️ 불균형 해결 방법 vs 모델 조합 실험...")
print("이것은 시간이 좀 걸릴 수 있습니다... ⏳")

# 결과 저장
all_results = {}
scaler = StandardScaler()

# 스케일링이 필요한 모델들
scaling_models = ['LogisticRegression', 'LogisticRegression_Balanced']

models = get_models()

for balance_method, balance_data in methods_data.items():
    print(f"\n🔄 {balance_method} 방법 테스트...")
    
    X_train = balance_data['X']
    y_train = balance_data['y']
    sample_weight = balance_data.get('sample_weight', None)
    
    # 스케일링
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test_final)
    
    for model_name, model in models.items():
        try:
            combination_name = f"{balance_method}_{model_name}"
            
            # 스케일링 데이터 선택
            if model_name in scaling_models:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test_final
            
            # XGBoost/LightGBM의 클래스 가중치 처리
            if 'Balanced' in model_name and ('XGBoost' in model_name or 'LightGBM' in model_name):
                if 'XGBoost' in model_name:
                    # XGBoost에서는 scale_pos_weight 사용
                    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                    model.set_params(scale_pos_weight=pos_weight)
                elif 'LightGBM' in model_name:
                    # LightGBM은 이미 class_weight='balanced'로 설정됨
                    pass
            
            # 모델 학습 (가중치가 있으면 적용)
            if sample_weight is not None and hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                model.fit(X_train_use, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train_use, y_train)
            
            # 예측
            test_pred = model.predict(X_test_use)
            test_proba = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 성능 계산
            accuracy = accuracy_score(y_test_processed, test_pred)
            precision = precision_score(y_test_processed, test_pred, zero_division=0)
            recall = recall_score(y_test_processed, test_pred, zero_division=0)
            f1 = f1_score(y_test_processed, test_pred, zero_division=0)
            
            # 불량률 예측 정확도
            actual_defect_rate = (y_test_processed == 1).sum() / len(y_test_processed) * 100
            predicted_defect_rate = (test_pred == 1).sum() / len(test_pred) * 100
            defect_rate_error = abs(actual_defect_rate - predicted_defect_rate)
            
            # 결과 저장
            all_results[combination_name] = {
                'balance_method': balance_method,
                'model': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'actual_defect_rate': actual_defect_rate,
                'predicted_defect_rate': predicted_defect_rate,
                'defect_rate_error': defect_rate_error,
                'train_size': len(X_train),
                'description': balance_data['description']
            }
            
            print(f"   ✅ {model_name:20s}: 정확도 {accuracy:.4f}, 불량률오차 {defect_rate_error:.2f}%p")
            
        except Exception as e:
            print(f"   ❌ {model_name}: 실패 - {str(e)}")
            continue

# 10. 📊 결과 분석 및 시각화
print(f"\n📊 종합 결과 분석...")

results_df = pd.DataFrame(all_results).T
results_df = results_df.sort_values('defect_rate_error', ascending=True)

print(f"\n🏆 불량률 예측 정확도 TOP 10:")
print("="*120)
print(f"{'순위':>2} {'불균형해결방법':15} {'모델':20} {'정확도':>8} {'정밀도':>8} {'재현율':>8} {'F1':>6} {'불량률오차':>8}")
print("-"*120)

for i, (combination, row) in enumerate(results_df.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['balance_method']:15s} {row['model']:20s} "
          f"{row['accuracy']:8.4f} {row['precision']:8.4f} {row['recall']:8.4f} "
          f"{row['f1']:6.4f} {row['defect_rate_error']:7.2f}%p")

# 11. 상세 시각화
fig, axes = plt.subplots(3, 3, figsize=(20, 18))

# 11-1. 불균형 해결 방법별 평균 성능
method_performance = results_df.groupby('balance_method').agg({
    'defect_rate_error': 'mean',
    'accuracy': 'mean',
    'f1': 'mean'
}).round(4)

ax1 = axes[0, 0]
method_names = method_performance.index
x_pos = np.arange(len(method_names))
bars = ax1.bar(x_pos, method_performance['defect_rate_error'], alpha=0.8, color='skyblue')
ax1.set_title('불균형 해결 방법별 평균 불량률 오차', fontweight='bold')
ax1.set_xlabel('불균형 해결 방법')
ax1.set_ylabel('불량률 오차 (%p)')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(method_names, rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)

# 11-2. 모델별 평균 성능
model_performance = results_df.groupby('model').agg({
    'defect_rate_error': 'mean',
    'accuracy': 'mean',
    'f1': 'mean'
}).round(4)

ax2 = axes[0, 1]
model_names = model_performance.index
x_pos = np.arange(len(model_names))
bars = ax2.bar(x_pos, model_performance['defect_rate_error'], alpha=0.8, color='lightgreen')
ax2.set_title('모델별 평균 불량률 오차', fontweight='bold')
ax2.set_xlabel('모델')
ax2.set_ylabel('불량률 오차 (%p)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_names, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

# 11-3. 불균형 해결 방법별 히트맵
ax3 = axes[0, 2]
pivot_data = results_df.pivot_table(values='defect_rate_error', 
                                   index='balance_method', 
                                   columns='model', 
                                   aggfunc='mean')
sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd_r', ax=ax3)
ax3.set_title('불균형방법 × 모델 불량률 오차', fontweight='bold')
ax3.set_xlabel('모델')
ax3.set_ylabel('불균형 해결 방법')

# 11-4. 정확도 vs 불량률 오차 스캐터
ax4 = axes[1, 0]
scatter = ax4.scatter(results_df['defect_rate_error'], results_df['accuracy'], 
                     c=results_df['f1'], cmap='viridis', s=50, alpha=0.7)
ax4.set_xlabel('불량률 오차 (%p)')
ax4.set_ylabel('정확도')
ax4.set_title('정확도 vs 불량률 오차', fontweight='bold')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='F1 Score')

# 11-5. TOP 5 조합 상세 비교
ax5 = axes[1, 1]
top_5 = results_df.head(5)
metrics = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(top_5))
width = 0.2

for i, metric in enumerate(metrics):
    ax5.bar(x + i*width, top_5[metric], width, label=metric.capitalize(), alpha=0.8)

ax5.set_title('TOP 5 조합 성능 비교', fontweight='bold')
ax5.set_xlabel('조합 (불균형방법_모델)')
ax5.set_ylabel('성능 점수')
ax5.set_xticks(x + width * 1.5)
ax5.set_xticklabels([f"{row['balance_method'][:8]}\n{row['model'][:12]}" 
                     for _, row in top_5.iterrows()], fontsize=8)
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 11-6. 불균형 해결 방법별 데이터 분포 변화
ax6 = axes[1, 2]
method_samples = []
method_labels = []
good_ratios = []
defect_ratios = []

for method, data in methods_data.items():
    if 'Weighted' not in method:  # 가중치 방법은 데이터 크기가 같으므로 제외
        y_counts = data['y'].value_counts().sort_index()
        total = len(data['y'])
        good_ratios.append(y_counts.get(0, 0) / total * 100)
        defect_ratios.append(y_counts.get(1, 0) / total * 100)
        method_labels.append(method)

x_pos = np.arange(len(method_labels))
width = 0.35

ax6.bar(x_pos - width/2, good_ratios, width, label='양품 비율', alpha=0.8, color='lightblue')
ax6.bar(x_pos + width/2, defect_ratios, width, label='불량품 비율', alpha=0.8, color='lightcoral')
ax6.set_title('불균형 해결 방법별 클래스 비율', fontweight='bold')
ax6.set_xlabel('불균형 해결 방법')
ax6.set_ylabel('비율 (%)')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(method_labels, rotation=45, ha='right')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# 11-7. 정밀도 vs 재현율 트레이드오프
ax7 = axes[2, 0]
scatter = ax7.scatter(results_df['recall'], results_df['precision'], 
                     c=results_df['defect_rate_error'], cmap='RdYlGn_r', s=50, alpha=0.7)
ax7.set_xlabel('재현율 (Recall)')
ax7.set_ylabel('정밀도 (Precision)')
ax7.set_title('정밀도 vs 재현율 (색상: 불량률 오차)', fontweight='bold')
ax7.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax7, label='불량률 오차 (%p)')

# 최고 성능 포인트 표시
best_idx = results_df['defect_rate_error'].idxmin()
best_recall = results_df.loc[best_idx, 'recall']
best_precision = results_df.loc[best_idx, 'precision']
ax7.scatter(best_recall, best_precision, color='red', s=100, marker='*', 
           label=f'최고성능\n({results_df.loc[best_idx, "balance_method"][:8]})')
ax7.legend()

# 11-8. 샘플링 방법 vs 가중치 방법 비교
ax8 = axes[2, 1]
sampling_methods = ['Original', 'Undersample', 'Oversample', 'Mixed', 'Synthetic']
weighting_methods = ['Weighted_Balanced', 'Weighted_Custom']

sampling_performance = results_df[results_df['balance_method'].isin(sampling_methods)].groupby('balance_method')['defect_rate_error'].mean()
weighting_performance = results_df[results_df['balance_method'].isin(weighting_methods)].groupby('balance_method')['defect_rate_error'].mean()

# 평균 성능 비교
sampling_avg = sampling_performance.mean()
weighting_avg = weighting_performance.mean()

categories = ['샘플링 방법\n평균', '가중치 방법\n평균']
values = [sampling_avg, weighting_avg]
colors = ['lightblue', 'lightgreen']

bars = ax8.bar(categories, values, color=colors, alpha=0.8)
ax8.set_title('샘플링 vs 가중치 방법 비교', fontweight='bold')
ax8.set_ylabel('평균 불량률 오차 (%p)')
ax8.grid(axis='y', alpha=0.3)

# 값 표시
for bar, val in zip(bars, values):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
            f'{val:.2f}%p', ha='center', va='bottom', fontweight='bold')

# 11-9. 최고 성능 조합의 혼동행렬
ax9 = axes[2, 2]
best_combination = results_df.index[0]
best_method = results_df.loc[best_combination, 'balance_method']
best_model_name = results_df.loc[best_combination, 'model']

# 최고 성능 조합으로 다시 학습하여 혼동행렬 생성
best_data = methods_data[best_method]
best_models = get_models()
best_model = best_models[best_model_name]

# 스케일링
if best_model_name in scaling_models:
    scaler_best = StandardScaler()
    X_train_best = scaler_best.fit_transform(best_data['X'])
    X_test_best = scaler_best.transform(X_test_final)
else:
    X_train_best = best_data['X']
    X_test_best = X_test_final

# 학습 및 예측
sample_weight_best = best_data.get('sample_weight', None)

# XGBoost/LightGBM 클래스 가중치 처리
if 'Balanced' in best_model_name and 'XGBoost' in best_model_name:
    pos_weight = (best_data['y'] == 0).sum() / (best_data['y'] == 1).sum()
    best_model.set_params(scale_pos_weight=pos_weight)

if sample_weight_best is not None and hasattr(best_model, 'fit') and 'sample_weight' in best_model.fit.__code__.co_varnames:
    best_model.fit(X_train_best, best_data['y'], sample_weight=sample_weight_best)
else:
    best_model.fit(X_train_best, best_data['y'])

best_pred = best_model.predict(X_test_best)
cm = confusion_matrix(y_test_processed, best_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax9,
            xticklabels=['양품(0)', '불량품(1)'], yticklabels=['양품(0)', '불량품(1)'])
ax9.set_title(f'최고성능 조합 혼동행렬\n{best_method} + {best_model_name}', fontweight='bold')
ax9.set_xlabel('예측')
ax9.set_ylabel('실제')

plt.tight_layout()
plt.suptitle('🎯 불균형 해결 방법 종합 비교 분석', fontsize=16, fontweight='bold', y=0.98)
plt.show()

# 12. 📋 상세 분석 및 인사이트
print(f"\n📋 상세 분석 및 인사이트")
print("="*60)

# 12-1. 최고 성능 조합 분석
best_3 = results_df.head(3)
print(f"🏆 TOP 3 불량률 예측 조합:")
for i, (combination, row) in enumerate(best_3.iterrows(), 1):
    print(f"   {i}. {row['balance_method']} + {row['model']}")
    print(f"      🎯 불량률 오차: {row['defect_rate_error']:.2f}%p")
    print(f"      📊 정확도: {row['accuracy']:.4f}")
    print(f"      🔧 훈련 데이터: {row['train_size']:,}개")

# 12-2. 불균형 해결 방법 효과 분석
print(f"\n📊 불균형 해결 방법별 효과:")
method_avg = method_performance.sort_values('defect_rate_error')
print(f"   🥇 최고: {method_avg.index[0]} (평균 오차: {method_avg.iloc[0]['defect_rate_error']:.2f}%p)")
print(f"   🥈 2위: {method_avg.index[1]} (평균 오차: {method_avg.iloc[1]['defect_rate_error']:.2f}%p)")
print(f"   🥉 3위: {method_avg.index[2]} (평균 오차: {method_avg.iloc[2]['defect_rate_error']:.2f}%p)")

# 원본 대비 개선 효과
original_avg = method_avg.loc['Original', 'defect_rate_error']
best_method_avg = method_avg.iloc[0]['defect_rate_error']
improvement = original_avg - best_method_avg
print(f"   📈 원본 대비 개선: {improvement:.2f}%p ({improvement/original_avg*100:.1f}% 개선)")

# 12-3. 모델별 효과 분석
print(f"\n🤖 모델별 불균형 대응 효과:")
model_avg = model_performance.sort_values('defect_rate_error')
print(f"   🎯 최고 모델: {model_avg.index[0]} (평균 오차: {model_avg.iloc[0]['defect_rate_error']:.2f}%p)")

# Balanced 버전과 일반 버전 비교
balanced_models = [m for m in model_avg.index if 'Balanced' in m]
regular_models = [m for m in model_avg.index if m.replace('_Balanced', '') in [b.replace('_Balanced', '') for b in balanced_models]]

print(f"   ⚖️ class_weight='balanced' 효과:")
for regular in regular_models:
    balanced = regular + '_Balanced'
    if balanced in model_avg.index:
        regular_score = model_avg.loc[regular, 'defect_rate_error'] if regular in model_avg.index else None
        balanced_score = model_avg.loc[balanced, 'defect_rate_error']
        if regular_score:
            improvement = regular_score - balanced_score
            print(f"      {regular}: {improvement:+.2f}%p ({'개선' if improvement > 0 else '악화'})")

# 12-4. 샘플링 vs 가중치 방법 결론
sampling_results = results_df[results_df['balance_method'].isin(sampling_methods)]
weighting_results = results_df[results_df['balance_method'].isin(weighting_methods)]

sampling_best = sampling_results['defect_rate_error'].min()
weighting_best = weighting_results['defect_rate_error'].min()

print(f"\n⚖️ 샘플링 vs 가중치 방법 결론:")
print(f"   📊 샘플링 방법 최고: {sampling_best:.2f}%p")
print(f"   🏋️ 가중치 방법 최고: {weighting_best:.2f}%p")
if sampling_best < weighting_best:
    print(f"   🏆 결론: 샘플링 방법이 {weighting_best - sampling_best:.2f}%p 더 우수")
else:
    print(f"   🏆 결론: 가중치 방법이 {sampling_best - weighting_best:.2f}%p 더 우수")

# 12-5. 데이터 크기별 효과 분석
print(f"\n📏 데이터 크기별 효과:")
size_analysis = results_df.groupby('balance_method').agg({
    'train_size': 'first',
    'defect_rate_error': 'mean'
}).sort_values('defect_rate_error')

for method, row in size_analysis.iterrows():
    if method != 'Original':
        original_size = size_analysis.loc['Original', 'train_size']
        size_change = row['train_size'] - original_size
        size_change_pct = size_change / original_size * 100
        print(f"   {method:15s}: 크기변화 {size_change:+6,}개 ({size_change_pct:+5.1f}%) → 오차 {row['defect_rate_error']:.2f}%p")

# 12-6. 최종 추천
print(f"\n💡 최종 추천:")
ultimate_best = results_df.iloc[0]
print(f"   🎯 최우선 추천: {ultimate_best['balance_method']} + {ultimate_best['model']}")
print(f"      📊 불량률 오차: {ultimate_best['defect_rate_error']:.2f}%p")
print(f"      📈 정확도: {ultimate_best['accuracy']:.4f}")
print(f"      🔧 F1-Score: {ultimate_best['f1']:.4f}")
print(f"      📝 설명: {ultimate_best['description']}")

# 실용적 관점에서의 추천
practical_threshold = 1.0  # 1%p 이하 오차
practical_candidates = results_df[results_df['defect_rate_error'] <= practical_threshold]

if len(practical_candidates) > 0:
    print(f"\n✨ 실용적 관점 (불량률 오차 {practical_threshold}%p 이하):")
    # 이 중에서 가장 간단한 방법 추천
    simplicity_order = ['Original', 'Weighted_Balanced', 'Weighted_Custom', 'Undersample', 'Oversample', 'Mixed', 'Synthetic']
    
    for simple_method in simplicity_order:
        simple_candidates = practical_candidates[practical_candidates['balance_method'] == simple_method]
        if len(simple_candidates) > 0:
            best_simple = simple_candidates.iloc[0]
            print(f"   🎯 간단한 추천: {best_simple['balance_method']} + {best_simple['model']}")
            print(f"      💼 장점: 구현이 간단하고 오차 {best_simple['defect_rate_error']:.2f}%p")
            break

print(f"\n💾 분석 결과 저장:")
print(f"   - all_results: 전체 실험 결과")
print(f"   - results_df: 정리된 결과 데이터프레임")
print(f"   - methods_data: 각 불균형 해결 방법별 데이터")
print(f"   - method_performance: 방법별 평균 성능")
print(f"   - model_performance: 모델별 평균 성능")

print(f"\n✅ 불균형 해결 방법 비교 분석 완료! 🎯")

# 13. 실제 적용 가이드
print(f"\n📚 실제 적용 가이드:")
print(f"="*50)
print(f"1️⃣ 빠른 적용: {ultimate_best['balance_method']} + {ultimate_best['model']}")
print(f"2️⃣ 데이터가 많으면: 언더샘플링 방법 고려")
print(f"3️⃣ 데이터가 적으면: 오버샘플링 또는 합성 샘플 고려") 
print(f"4️⃣ 구현이 간단해야 하면: class_weight='balanced' 옵션 활용")
print(f"5️⃣ 불량률이 매우 낮으면: 가중치 방법이 더 안전할 수 있음")
print(f"6️⃣ 실시간 예측이 필요하면: 가벼운 모델 + 간단한 불균형 해결 방법")

print(f"\n🎉 모든 분석이 완료되었습니다!")