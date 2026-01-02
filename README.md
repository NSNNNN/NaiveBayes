
# 🎯 دیتاست انتخابی: **Breast Cancer Dataset**  
(یکی از دیتاست‌های کلاسیک برای Classification)

در ادامه:

- مدل Naive Bayes را آموزش می‌دهیم  
- پیش‌بینی می‌کنیم  
- شاخص‌های ارزیابی را محاسبه می‌کنیم:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  

---

# 🧠 **کد کامل اجرای Naive Bayes روی دیتاست Breast Cancer**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

---

# 📊 **نتایج معمول (تقریباً مشابه اجرای واقعی)**

| شاخص | مقدار تقریبی |
|------|--------------|
| Accuracy | 0.94 |
| Precision | 0.95 |
| Recall | 0.97 |
| F1-score | 0.96 |

---

# 🧩 **ماتریس درهم‌ریختگی (Confusion Matrix)**

به‌طور معمول چیزی شبیه این می‌شود:

|        | Pred 0 | Pred 1 |
|--------|--------|--------|
| **True 0** | 39 | 4 |
| **True 1** | 2 | 69 |

---

# 🌱 **تحلیل نتایج**

- Naive Bayes روی این دیتاست عملکرد **بسیار خوب** دارد  
- Recall بالا نشان می‌دهد مدل موارد مثبت (سرطان) را خوب تشخیص می‌دهد  
- Precision بالا یعنی خطای مثبت کاذب کم است  
- Confusion Matrix نشان می‌دهد مدل اشتباهات کمی دارد  

---
