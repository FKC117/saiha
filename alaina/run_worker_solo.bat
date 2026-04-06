@echo off
echo Starting ChatFlow Celery Worker (Solo Mode)...
REM Activate virtual environment
call ..\venv\Scripts\activate
REM Run Celery
celery -A alaina worker --loglevel=info -P solo
pause
