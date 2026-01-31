📁 โครงสร้างก่อนเริ่ม
Detection-Project/
│
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   │
│   └── processed/
│
├── notebooks/
├── src/
├── models/
├── output/
├── requirements.txt
└── main.py
🔹 STEP 1: เข้าโฟลเดอร์โปรเจค
เปิด Terminal / VS Code
cd Detection-Project
ตรวจสอบว่าอยู่ถูกที่
ls
ต้องเห็น
data  notebooks  src  main.py  requirements.txt
🔹 STEP 2: สร้าง Virtual Environment (แนะนำมาก)
Windows
python -m venv venv
venv\Scripts\activate
macOS / Linux
python3 -m venv venv
source venv/bin/activate
ถ้าสำเร็จจะเห็น
(venv)
🔹 STEP 3: ติดตั้ง Library
pip install -r requirements.txt
รอจนเสร็จ
🔹 STEP 4: เตรียม Dataset
นำไฟล์จาก Kaggle:
creditcard.csv
แล้ว เปลี่ยนชื่อเป็น
dataset.csv
ใส่ไว้ที่
data/raw/dataset.csv
⚠️ ห้ามแก้ไฟล์ CSV
🔹 STEP 5: (ทางเลือก) รัน Notebook
ถ้าอยากดูขั้นตอนการคิด
jupyter notebook
เปิดตามลำดับ
1️⃣ 01_exploration.ipynb
2️⃣ 02_preprocessing.ipynb
3️⃣ 03_model_training.ipynb
4️⃣ 04_evaluation.ipynb
📌 Notebook = วิเคราะห์ / ทดลอง
📌 ไม่จำเป็นต้องรันครบก็ได้
🔹 STEP 6: รันระบบจริงทั้งหมด (สำคัญ)
รันไฟล์เดียวจบทั้งโปรเจค 👇
python main.py
ถ้าถูกต้อง จะเห็น
Training completed successfully
🔹 STEP 7: ดูผลลัพธ์
📁 models/
model.pkl
→ โมเดลที่ train แล้ว
📁 output/metrics/
evaluation.txt
ภายในจะมี
precision
recall
f1-score
confusion matrix