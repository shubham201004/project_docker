from streamlit import fragment, subheader, markdown

fragment()
def compute_welcome_tab():
    markdown("""
---
Welcome to the **Plant Disease Prediction System**, a deep-learning-powered tool designed to identify diseases in plants using **Convolutional Neural Networks (CNNs)**. This platform helps farmers and researchers detect plant diseases early and take necessary actions to improve crop health.  

#### **🔍 How It Works:**  
1. **Upload an Image 🌱**  
   - Choose an image of a plant leaf affected by a potential disease. Supported formats: **JPG, JPEG, PNG**.  

2. **Preprocessing 🖼️**  
   - The image is resized to **128x128 pixels** to match the model's input size.  

3. **Model Prediction 🧠**  
   - A trained **TensorFlow Keras CNN model** analyzes the image and predicts the disease category.  

4. **Results & Insights 📊**  
   - The model outputs the **most likely disease** from a list of 38 plant diseases, including **healthy** conditions.  

5. **AI-Powered Suggestions 💡** *(Optional: If using OpenAI for recommendations)*  
   - The system can generate treatment and prevention tips for the detected disease.  

---

### **🚀 Features:**  
✅ **Fast & Accurate CNN Model** for plant disease detection  
✅ **Supports Various Crops**: Apples, Grapes, Tomatoes, Potatoes, Corn, and more  
✅ **User-Friendly Interface** powered by **Streamlit**  
✅ **Secure & Efficient Image Processing**  

Start detecting plant diseases now and take proactive steps for a healthier harvest! 🌾  

---

Let me know if you'd like any modifications! 🚀
    """, unsafe_allow_html=True)

if __name__ == '__page__':
    compute_welcome_tab()