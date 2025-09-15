# **DataPulse: Automated Data Analysis Platform** 🚀

**Developed by Maharshi Tripathi**  
👤 [LinkedIn](https://www.linkedin.com/in/maharshi-tripathi-26b64a222)  

---

**DataPulse** is a powerful, **Streamlit**-based data analysis platform designed to **streamline** the end-to-end workflow from data ingestion to professional reporting. Built with **efficiency** and **user experience** in mind, it leverages **AI-driven insights** and **advanced analytics** to empower users to uncover **actionable insights** from their data effortlessly. 🔍💡

## 🚀 **Overview**
DataPulse automates complex data workflows, making it ideal for **analysts**, **businesses**, and **researchers**. It supports **multi-file uploads**, **intelligent preprocessing**, **domain-specific analysis suggestions**, **parallel execution**, **interactive dashboards**, and **professional PDF reports**. With a focus on **privacy** and **performance**, it ensures **secure**, **client-side processing** and **fast results**. 🛡️⚡

## ✨ **Key Features**

- **Seamless Data Ingestion**: Drag-and-drop support for **CSV/Excel** files with automatic merging for multi-file datasets. 📁➡️📊
- **Smart Preprocessing**: Handles **missing values**, **outliers**, **scaling**, and **sensitive data anonymization** (e.g., emails, SSNs). 🔧🛠️
- **AI-Powered Suggestions**: Detects data domains (**time series**, **numeric**, **categorical**) to recommend tailored analyses. 🤖🎯
- **Parallel Analysis Execution**: Runs **multiple analyses** concurrently (e.g., descriptive stats, anomaly detection, clustering) with **real-time progress tracking**. ⚡📈
- **Dynamic Dashboards**: Interactive visualizations using **Plotly**, with filters and **AI-generated insights** powered by local LLMs (Ollama). 📊🔍
- **Professional Reporting**: Exports comprehensive **PDF reports** with embedded charts and concise summaries. 📄📋
- **Demo Mode**: Test with sample datasets to explore features without uploading data. 🎮🧪

## 🛡️ **Privacy & Security**

- **Client-Side Processing**: Ensures data stays **in-memory** for small datasets, enhancing **privacy**. 🔒💾
- **PII Anonymization**: Automatically hashes sensitive fields (e.g., emails, phone numbers) using **SHA-256**. 🛡️🔐
- **Secure Configuration**: Manages **API keys** via Streamlit's **secrets.toml** for safe integration with external services. 🔑💳

## 🛠️ **Technical Stack**

- **Frontend**: Streamlit for responsive, user-friendly interfaces. 🌐🎨
- **Backend**: Python 3.8+, **Pandas**, **NumPy**, **Scikit-learn** for data processing and machine learning. 🐍⚙️
- **Visualization**: **Plotly** for interactive charts; **ReportLab** for PDF generation. 📊🖼️
- **AI Integration**: **Ollama** for local, low-latency insight generation (e.g., **Llama 3.2** for summaries). 🤖⚡
- **Dependencies**: Listed in **requirements.txt** (includes streamlit, pandas, ollama, plotly, etc.). 📦

## 📦 **Installation**

### Prerequisites
- Python **3.8** or higher 🐍
- **Ollama** installed for local AI inference ([download here](https://ollama.ai)) 🤖
- **4GB+ RAM** for optimal performance (**8GB+** recommended for larger datasets) 💾

### Steps
1. **Clone the Repository**:
   ```
   git clone https://github.com/your-username/datapulse.git
   cd datapulse
   ```
   📥

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
   🛠️

3. **Set Up Ollama**:
   - Install Ollama and pull a model:
     ```
     ollama pull llama3.2:3b
     ollama serve
     ```
   - Ensure the server runs at `http://localhost:11434`.
     🚀

4. **Run the Application**:
   ```
   streamlit run main.py
   ```
   🎉

## 🚀 **Usage**

- **Upload Data**: Drag-and-drop **CSV/Excel** files or use **demo mode** with sample sales data. 📤🗂️
- **Preprocess**: Automatically clean and prepare data (handles **duplicates**, **missing values**, **outliers**). 🔄⚗️
- **Analyze**: Select from **AI-suggested analyses** (e.g., time series, regression, clustering). 📊
- **Visualize**: Explore **interactive dashboards** with filters and **AI-generated insights**. 👁️‍🗨️
- **Export**: Generate **professional PDF reports** with visualizations and summaries. 📤📄

**Performance Note**: Optimized for datasets up to **10,000 rows**. For larger datasets, expect minor delays (addressed with data sampling and local AI). ⚡⏱️

## 🌟 **Example Workflow**

1. Upload a **sales dataset** (e.g., 100 rows of dates, sales, categories). 📊
2. Preprocess to **scale numeric columns** and **impute missing values**. 🔄
3. Select analyses like **"Correlation Analysis"** and **"Anomaly Detection."** 📈
4. View **dynamic dashboard** with scatter plots and **AI insights** (e.g., "Sales spike in Q2, review pricing strategy"). 📉💡
5. Export a **PDF report** with embedded charts and a concise summary. 📋✅

## 🛠️ **Deployment**
Deploy DataPulse on cloud platforms for broader access:
- **Streamlit Community Cloud**: Share via [share.streamlit.io](https://share.streamlit.io). ☁️🌍
- **Heroku/AWS/Azure**: Configure with secrets management for secure API handling. 🏗️🔒
- **Local Hosting**: Run on a local server for private, **high-performance** use. 💻🏠

## 📚 **Future Enhancements**

- Integration with **cloud databases** (e.g., **Google BigQuery**) for real-time data. ☁️📊
- **Advanced anomaly detection** with streaming data support. 🚨📡
- Customizable **dashboard themes** and export formats (e.g., **PPTX**, **HTML**). 🎨📤

## 📬 **Contact**
For **feedback**, **contributions**, or **inquiries**, connect with me:  

**Maharshi Tripathi**  
👤 [LinkedIn](https://www.linkedin.com/in/maharshi-tripathi-26b64a222)  
📧 Email:  maharshitripathi302@gmail.com  

## 🙏 **Acknowledgments**
**Built with passion by Maharshi Tripathi**, an MCA student passionate about **data analytics** and **AI-driven solutions**. Inspired by **real-world needs** for automated, **user-friendly data analysis tools**. ❤️🤝

**DataPulse: Empowering Insights, One Dataset at a Time**  
Powered by **Streamlit** and **Ollama** for **fast**, **secure analytics**. 🌟🚀
