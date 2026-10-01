Installation and Run:
=====================
~~~powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe brno_heat_landsat.py
.\venv\Scripts\python.exe -m unittest discover -s tests -v
~~~