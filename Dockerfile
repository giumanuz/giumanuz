FROM python:3.9-slim

RUN apt-get update && apt-get install -y cron git


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY crontab /etc/cron.d/daily-cron
COPY . /app

# Imposta i permessi per la crontab
RUN chmod 0644 /etc/cron.d/daily-cron && crontab /etc/cron.d/daily-cron

# Crea un file di log
RUN touch /var/log/cron.log

# Avvia cron in foreground e segui i log
CMD ["bash", "-c", "rm -f /var/run/crond.pid && service cron start && tail -f /var/log/cron.log"]
