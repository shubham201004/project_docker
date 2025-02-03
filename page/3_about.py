from streamlit import fragment, subheader, markdown

fragment()
def compute_about():
    markdown("""
---             
We are a team of five enthusiastic students who developed the **Plant Disease Prediction** application using **Deep Learning**. Our goal is to provide an easy-to-use tool that helps farmers and researchers **detect plant diseases** early, ensuring healthier crops and better yields.  

### **👨‍💻 Our Team:**  
🌟 **Akshata**  
🌟 **Amit**  
🌟 **Saloni**  
🌟 **Shubham**  
🌟 **Gaurav**  

### **🎯 Our Mission:**  
We aim to harness **AI and Computer Vision** to assist in **early disease detection**, reducing crop losses and improving agricultural productivity.  

### **🛠️ Technologies Used:**  
✅ **TensorFlow & Keras** – Deep Learning Model  
✅ **Streamlit** – Interactive Web App  
✅ **OpenAI & Langchain** – AI-based Disease Suggestions  
✅ **Docker** – Containerized Deployment  

### **📢 Acknowledgment:**  
We extend our gratitude to **CDAC** for their guidance and support in bringing this project to life.  

🌿 **Empowering Agriculture with AI!** 🌿  

---
    """, unsafe_allow_html=True)

if __name__ == '__page__':
    compute_about()
