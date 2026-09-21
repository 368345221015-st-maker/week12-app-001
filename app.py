# app.py
# แอป Streamlit สำหรับโหลดโมเดล Titanic (.joblib) แล้วทำนายผลจากข้อมูลที่ผู้ใช้กรอก
# วางไฟล์นี้ไว้ในโฟลเดอร์เดียวกับไฟล์โมเดล แล้วรันด้วยคำสั่ง: streamlit run app.py

import streamlit as st
import joblib
import numpy as np
import sklearn

# ชื่อไฟล์โมเดล (ไฟล์จริงที่อัปโหลดมาชื่อ titanic_tree.joblib)
# หากเปลี่ยนชื่อไฟล์โมเดล ให้แก้ค่านี้ที่เดียว
MODEL_FILENAME = "titanic_tree.joblib"


@st.cache_resource  # แคชโมเดลไว้ในหน่วยความจำ ไม่ต้องโหลดใหม่ทุกครั้งที่ผู้ใช้กดปุ่ม
def load_model(filename: str):
    """โหลดโมเดลจากไฟล์ joblib คืนค่า (model, error_message)"""
    try:
        model = joblib.load(filename)
        return model, None
    except FileNotFoundError:
        return None, f"ไม่พบไฟล์ '{filename}' กรุณาตรวจสอบว่าไฟล์นี้อยู่ในโฟลเดอร์เดียวกับ app.py"
    except Exception as e:
        return None, (
            f"โหลดโมเดลไม่สำเร็จ: {e}\n"
            f"เวอร์ชัน scikit-learn ในเครื่องนี้คือ {sklearn.__version__} "
            "ลองตรวจสอบว่าตรงกับตอนที่บันทึกโมเดลหรือไม่"
        )


# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ทำนายการรอดชีวิตบนไททานิค", page_icon="🚢")
st.title("🚢 ทำนายการรอดชีวิตบนไททานิค")
st.write("กรอกข้อมูลผู้โดยสารด้านล่าง แล้วกดปุ่มเพื่อทำนายผล")

# โหลดโมเดล
model, error = load_model(MODEL_FILENAME)

if error:
    # ถ้าโหลดโมเดลไม่สำเร็จ ให้แสดงข้อความแจ้งเตือนและหยุดการทำงานของแอป
    st.error(error)
    st.stop()

# แสดงฟอร์มรับข้อมูลจากผู้ใช้ ให้ตรงกับฟีเจอร์ที่โมเดลต้องใช้:
# Pclass, Sex_female, Age, Fare, FamilySize
with st.form("prediction_form"):
    pclass = st.selectbox("ชั้นโดยสาร (Pclass)", options=[1, 2, 3], index=2)
    sex = st.radio("เพศ", options=["หญิง", "ชาย"])
    age = st.number_input("อายุ (Age)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    fare = st.number_input("ค่าโดยสาร (Fare)", min_value=0.0, value=32.0, step=1.0)
    family_size = st.number_input(
        "จำนวนสมาชิกครอบครัวที่โดยสารมาด้วย (FamilySize)",
        min_value=0, max_value=20, value=0, step=1,
    )
    submitted = st.form_submit_button("ทำนายผล")

if submitted:
    # แปลงค่า "เพศ" ให้เป็น Sex_female (1 = หญิง, 0 = ชาย) ตามที่โมเดลถูกเทรนมา
    sex_female = 1 if sex == "หญิง" else 0

    # จัดเรียงลำดับฟีเจอร์ให้ตรงกับตอนเทรนโมเดล
    features = np.array([[pclass, sex_female, age, fare, family_size]])

    # ทำนายผลและความน่าจะเป็น
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    st.subheader("ผลการทำนาย")
    if prediction == 1:
        st.success(f"🟢 คาดว่า **รอดชีวิต** (ความน่าจะเป็น {proba[1]*100:.1f}%)")
    else:
        st.error(f"🔴 คาดว่า **ไม่รอดชีวิต** (ความน่าจะเป็น {proba[0]*100:.1f}%)")

    with st.expander("ดูรายละเอียดข้อมูลที่ส่งเข้าโมเดล"):
        st.write(
            {
                "Pclass": pclass,
                "Sex_female": sex_female,
                "Age": age,
                "Fare": fare,
                "FamilySize": family_size,
            }
        )
