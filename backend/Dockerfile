FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk;nltk.download('stopwords'); nltk.download('punkt_tab');"
RUN mv -f ~/nltk_data /usr/local/nltk_data

COPY . .

CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8080"]
