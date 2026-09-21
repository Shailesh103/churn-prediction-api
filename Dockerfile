# 1. Base image: chhota Linux jisme Python 3.13 pehle se installed hai (tumhare Python jaisa)
FROM python:3.13-slim

# 2. Container ke andar kaam karne ka folder. Aage saari commands yahin chalengi
WORKDIR /app

# 3. Pehle SIRF requirements.txt copy karo (poora code nahi, taaki layer cache ka fayda mile)
COPY requirements.txt .

# 4. Packages install karo. --no-cache-dir: temporary files mat rakho (image chhoti rahegi)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Ab code aur model copy karo
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# 6. Batao ki API 8000 port pe chalegi (sirf documentation ke liye)
EXPOSE 8000

# 7. Container start hote hi ye command chalao.
# --host 0.0.0.0 zaroori hai: warna container ke bahar se API nahi khulegi
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]