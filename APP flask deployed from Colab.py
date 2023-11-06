!pip install pyngrok
!pip install flask_ngrok
!pip install flask_cors
!pip install Flask-Login
!pip install Flask-SQLAlchemy
!pip install Flask-Paginate

from flask_login import current_user
from pyngrok import ngrok
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, jsonify, Response, flash
from werkzeug.utils import secure_filename
from flask import send_file
from flask import Flask, redirect, url_for, request, render_template
from flask_cors import CORS
from flask_paginate import Pagination, get_page_args
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_sqlalchemy import SQLAlchemy
import json
import math
import pandas as pd


app = Flask(__name__, template_folder='/content/drive/MyDrive/templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////content/drive/MyDrive/database.db'  # Database URI
app.config['SECRET_KEY'] = 'secretkey'  # Set a secret key for session management

CORS(app)
db = SQLAlchemy(app)


# Configure Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'


ngrok.set_auth_token("Put the auth token")
public_url =  ngrok.connect(port_no).public_url

# User class for Flask-Login
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)

# User loader function for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        # Check if the username already exists
        if User.query.filter_by(username=username).first():
            flash('User already exists', 'error')
        elif User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
        else:
            new_user = User(username=username, password=password, email=email)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
    return render_template('signup.html')


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

# Logout route
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route("/", methods=["GET"])
def home():
   if current_user.is_authenticated:
      # Pagination code
      page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
      total_rows = len(data['userName'])
      num_pages = math.ceil(total_rows / per_page)
      pagination_data = {key: data[key][offset: offset + per_page].tolist() for key in data}  # Convert Series to list
      pagination = Pagination(page=page, per_page=per_page, total=total_rows)
      return render_template('template.html', data=pagination_data, pagination=pagination, num_pages=num_pages, current_time=current_time)
   else:
        return redirect(url_for('login'))
   
   !sudo allow 5000

# Create database tables
with app.app_context():
    db.create_all()

run_with_ngrok(app)
if __name__ == "__main__":
    app.run()
