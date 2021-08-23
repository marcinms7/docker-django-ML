# docker-django-ML
Dockerising and optimising my Machine Learning Django web app  

To run:
### `docker-compose up`

Runs on http://0.0.0.0:8000/ host

Docker will build super user/admin, that you can access at http://0.0.0.0:8000/admin 
You can set up your admin in autoadmin.py
Default is user:admin password:pass 

For tests you can use seed_data.csv file from the main folder (it's https://www.kaggle.com/rwzhang/seeds-dataset with adjusted Target column).


