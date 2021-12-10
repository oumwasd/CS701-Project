# CS701 Project of Group 7
Python 3.9+\
R 4.1.2+\
Python ไลบรารี่ที่ต้องใช้\
scikit-learn\
hmeasure\
pandas\
pylint\
imbalanced-learn\
ipykernel\
xgboost\
R ไลบรารี่ที่ต้องใช้\
tidyverse\
devtools\
scmamp\
สามารถใช้ poetry เพื่อติดตั้งไลบารี่สำหรับ Python ได้ (ไฟล์ pyproject.toml จำเป็นสำหรับการใช้งาน poetry)\
สามารถ run แต่ละไฟล์ที่มีชื่ออัลกอริทึมในโฟลเดอร์ algro (ตัวอย่างเช่น elasticnet_logistic_regression_smote.py)\
เมื่อ run เสร็จแล้วจะผลลัพธ์จะอยู่ในโฟลเดอร์ result\
ถ้าต้องการนำไฟล์ผลลัพธ์มาทดสอบทางสถิติจะต้องใช้ภาษา R โดยต้องติดตั้งไลบรารี่ scmamp ด้วยคำสั่ง devtools::install_github("b0rxa/scmamp")